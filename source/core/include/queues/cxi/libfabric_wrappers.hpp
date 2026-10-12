#ifndef ST_CXI_QUEUE_LIBFABRIC_WRAPPER
#define ST_CXI_QUEUE_LIBFABRIC_WRAPPER

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_trigger.h>

#include <cstdint>
#include <map>
#include <vector>
// clang-format off
#include <rdma/fi_cxi_ext.h>
// clang-format on

#include "misc/print.hpp"
#include "safety/gpu.hpp"
#include "safety/libfabric.hpp"
#include "safety/mpi.hpp"

/**
 * @file libfabric_wrappers.hpp
 * @brief File for abstractions on top of libfabric.
 */

/**
 * @brief An object that manages most of the background libfabric objects.
 * @details This class is designed to hide away most of the set up and teardown of the
 * various libfabric objects needed to support communication. With the current
 * implementation, only libfabric memory regions, counters, and address are available to
 * be obtained through this class.
 *
 * The use of libfabric is tied to the needs of the CXI libfabric provider at the moment.
 * It does attempt to select a specific nic based on the current GPU device assigned to or
 * selected by the process. The exact process depends on the GPU model involved (aka if
 * USE_TIOGA is defined).
 *
 * Generally designed to be a single instance at a time, but that's more of an abstract
 * restriction than a software limitation. Hardware limits (counters, deferred work queue
 * entries, etc.) may come into play if multiple are used at the same time in the same
 * run.
 *
 * The libfabric code here is likely an amalgamation of novice knowledge, fixes from
 * version upgrades, and provider/design specific choices. Take care when changing flags
 * or altering code.
 *
 */
class LibfabricInstance
{
public:
    /** @brief Default constructor since nothing can actually be made yet. */
    LibfabricInstance() = default;
    /** @brief Destructor that cleans up all libfabric objects managed by this class.
     * @details Does not clean up objects that were handed out (such as memory regions),
     * so if something is not freed that would cause libfabric to be unable to free
     * another resource, an error message is printed. Order of freeing: endpoint, progress
     * counter, address vector, receive queue, transmit queue, domain, and then the
     * fabric.
     */
    ~LibfabricInstance();

    /** @brief Copy constructor deleted. */
    LibfabricInstance(const LibfabricInstance&) = delete;
    /** @brief Copy assignment operator deleted. */
    LibfabricInstance& operator=(const LibfabricInstance&) = delete;

    /**
     * @brief The true "constructor" for this class. Collective operation.
     * @details This method first creates all of the necessary local libfabric objects
     * before initiating the exchange to get the address of all other peers (using MPI).
     * @param [in] comm The MPI communicator to perform the allgather on for the exchange
     * of CXI addresses.
     */
    void initialize(MPI_Comm comm)
    {
        initialize_libfabric(comm);
        initialize_peer_addresses(comm);
    }

    /**
     * @brief Create a libfabric counter, with the option to note to check it when making
     * progress.
     * @details Libfabric counters are created with the basic settings
     * (FI_CNTR_EVENTS_COMP, FI_WAIT_UNSPEC) on LibfabricInstance::domain. If @p dwq_track
     * is true, then it is added to LibfabricInstance::dwq_progress_counters.
     * @param [in] dwq_track Indicate if the allocated libfabric counter should be checked
     * when this class is asked to make progress on libfabric-based communications.
     * @return A pointer to newly allocated libfabric fid_cntr struct.
     * @sa LibfabricInstance::progress_dwq()
     * @sa LibfabricInstance::dealloc_counter()
     */
    struct fid_cntr* alloc_counter(bool dwq_track)
    {
        struct fid_cntr*    new_ctr;
        struct fi_cntr_attr cntr_attr = {
            .events   = FI_CNTR_EVENTS_COMP,
            .wait_obj = FI_WAIT_UNSPEC,
        };
        force_libfabric(fi_cntr_open(domain, &cntr_attr, &new_ctr, NULL));
        if (dwq_track)
        {
            dwq_progress_counters[new_ctr] = 0;
        }
        return new_ctr;
    }

    /**
     * @brief Return a libfabric counter back to be deallocated.
     * @details If this counter was allocated with option to be tracked for progress
     * (indicated by its presence in LibfabricInstance::dwq_progress_counters), then two
     * more progress checks are made before attempting to remove it. This is to try and
     * ensure that any outstanding progress counter updates have been accounted for. The
     * second call is from manual testing of the libfabric code, and may prove to be
     * unnecessary later.
     *
     * While this code doesn't necessarily check for this, it goes without saying that the
     * counter should not be in use by any deferred work queue items when it is
     * deallocated.
     * @param [in] counter The pointer to the counter object to be closed.
     * @sa LibfabricInstance::alloc_counter()
     */
    void dealloc_counter(struct fid_cntr* counter)
    {
        if (dwq_progress_counters.contains(counter))
        {
            /* Progress twice based on Whit's findings */
            progress_dwq();
            progress_dwq();
            /* Remove it from available progress counters */
            dwq_progress_counters.erase(counter);
        }
        force_libfabric(fi_close(&counter->fid));
    }

    /**
     * @brief Allocate and enable a libfabric memory region.
     * @details This function is a wrapper for two libfabric actions: allocating a memory
     * region and enabling it. Currently this library has no need to do any actions
     * between those two steps, so this function is a convenience wrapper to make higher
     * levels of code more readable.
     * @param [in] buffer The buffer address to associate with the memory region.
     * @param [in] len The size of the buffer, in bytes.
     * @param [in] access Any access flags to pass to fi_mr_reg.
     * @param [in] flags Any other flags to pass to fi_mr_reg.
     * @return A pointer to the new memory region that has been allocated.
     */
    struct fid_mr* create_mr(const void* buffer, size_t len, uint64_t access,
                             uint64_t flags)
    {
        // Enable MR
        struct fid_mr* new_mr = alloc_mr(buffer, len, access, flags);
        force_libfabric(fi_mr_enable(new_mr));
        Print::out("Buffer:", buffer, "has mr key: ", fi_mr_key(new_mr));
        return new_mr;
    }

    /**
     * @brief Allocate, bind a counter to, and enable a libfabric memory region.
     * @details Like LibfabricInstance::create_mr(), but this version binds a counter to
     * the memory region before enabling it. This function does not allocate the counter
     * for the user, it must be passed in.
     * @param [in] buffer The buffer address to associate with the memory region.
     * @param [in] len The size of the buffer, in bytes.
     * @param [in] access Any access flags to pass to fi_mr_reg.
     * @param [in] alloc_flags Any other flags to pass to fi_mr_reg.
     * @param [in] counter The libfabric counter to bind the memory region.
     * @param [in] bind_flags What actions the libfabric counter should track.
     * @return A pointer to the newly allocated memory region that has been enabled and @p
     * counter has been bound to.
     */
    struct fid_mr* create_mr_with_counter(const void* buffer, size_t len, uint64_t access,
                                          uint64_t alloc_flags, fid_cntr* counter,
                                          uint64_t bind_flags)
    {
        struct fid_mr* new_mr = alloc_mr(buffer, len, access, alloc_flags);
        force_libfabric(fi_mr_bind(new_mr, &(counter)->fid, bind_flags));
        force_libfabric(fi_mr_enable(new_mr));
        Print::out("Buffer:", buffer, "has mr key: ", fi_mr_key(new_mr));
        return new_mr;
    }

    /**
     * @brief Perform a look up in the address vector to get a specific rank's address.
     * @param [in] rank The MPI rank (of ::MPIS_COMM_WORLD) to lookup.
     * @return The fi_addr_t of the peer rank.
     * @sa LibfabricInstance::peers
     */
    fi_addr_t get_peer(int rank)
    {
        return peers.at(rank);
    }

    /**
     * @brief Queue up a deferred work queue entry to libfabric.
     * @details Because libfabric will return an error code when there are no more
     * deferred work queue slots available, this function (and class) must manage
     * availability. This means that if there are no slots available, then the function
     * will block while making progress until a slot has been freed up.
     *
     * Because this class uses software control of the number of deferred work queue
     * slots, if it gets it wrong, an exception will be thrown. The current number of
     * slots (LibfabricInstance::MAX_DWQ_SLOTS) is set to 254 from manual testing. It
     * seems like the true limit should be controllable from some layer of the CXI stack,
     * but this value is the most stable. Note that on some systems, setting specific
     * values with the job scheduler may be necessary to have the hardware actually
     * allocate enough slots, and this value may change with different systems.
     *
     * Each successful deferred work entry that is given to libfabric consumes one slot.
     * The completion counter used for each deferred work queue entry MUST have been
     * created with LibfabricInstance::alloc_counter() with @p dwq_track set to "true".
     * Otherwise the entry will consume a slot that can never be returned.
     *
     * @param [in] work_entry The deferred work queue entry to hand to libfabric.
     * @sa LibfabricInstance::progress_dwq()
     */
    void queue_work(struct fi_deferred_work* work_entry)
    {
        Print::out("<H> Threshold:", work_entry->threshold, work_entry->triggering_cntr,
                   work_entry->completion_cntr);

        while (dwq_slots_used == MAX_DWQ_SLOTS)
        {
            progress_dwq();
        }
        force_libfabric(fi_control(&domain->fid, FI_QUEUE_WORK, work_entry));
        dwq_slots_used++;
    }

    /** @brief Make libfabric progress by checking all progress counters.
     * @details The design behind this function is from testing with libfabric
     * version 2.3.1 and older. It seems that to truly get the completion counters
     * associated with deferred work queue entries to update, an interaction with the
     * receive (completion) queue in libfabric must be made. This is done via reading the
     * value of LibfabricInstance::progress_ctr. Binding this to an endpoint with FI_RECV
     * (LibfabricInstance::initialize_libfabric()) makes reading it cause this "completion
     * queue" update to happen, even though it will never increase from deferred work
     * queue receive operations completing.
     *
     * After that check, the function iterates over all deferred work queue progress
     * counters to see if any of them have a new value different from their last read. If
     * they do, that means the difference between the current value and the last read is
     * the number of new deferred work queue slots that have been completed and are now
     * open again.
     *
     * Currently, the design of this function relies on DWQ completion counters not being
     * reset and always strictly increasing.
     */
    void progress_dwq()
    {
        /* Progress regular counter first. */
        fi_cntr_read(progress_ctr);
        uint64_t freed_slots = 0;
        for (auto& [counter, last_value] : dwq_progress_counters)
        {
            uint64_t new_value = fi_cntr_read(counter);
            freed_slots += (new_value - last_value);
            last_value = new_value;
        }

        if (freed_slots)
        {
            Print::out("Freed:", freed_slots, dwq_slots_used);
        }

        dwq_slots_used -= freed_slots;
        /* Read again */
        fi_cntr_read(progress_ctr);
    }

    /** @brief Provider's data and features */
    struct fi_info* fi;
    /** @brief Represents the network */
    struct fid_fabric* fabric;
    /** @brief A subsection of the network */
    struct fid_domain* domain;
    /** @brief Address vector for connections */
    struct fid_av* av;
    /** @brief The main endpoint for the CXI provider */
    struct fid_ep* ep;
    /** @brief The transmit completion queue (created, but is unused) */
    struct fid_cq* txcq;
    /** @brief The receive completion queue */
    struct fid_cq* rxcq;
    /** @brief Key counter for driving progress in libfabric. */
    struct fid_cntr* progress_ctr;

private:
    /**
     * @brief The function that controls the logic for selecting the NIC to use for this
     * process.
     * @details NIC selection logic depends on system, but for all current test cases it
     * is the GPU that is closest to the NIC. Note that in the CXI libfabric provider,
     * each NIC shows up as its own provider. LibfabricInstance::initialize_libfabric()
     * will get all available NICs when it "asks for a CXI provider", so this function is
     * necessary to narrow down which fabric is the one we want to use.
     * @param [in, out] fi_info_ptr The list of libfabric providers that match the
     * necessary requirements (should just be CXI providers). Since it is a linked list,
     * it will be set to the correct provider upon returning from this function.
     */
    void select_fi_nic(fi_info*& fi_info_ptr);

    /**
     * @brief Prepare all necessary libfabric objects that are needed to stand up a CXI
     * libfabric provider.
     * @details Most of the implementation of this function uses standard libfabric calls
     * to set everything up. The values for the "hints" that are passed to fi_getinfo()
     * are designed to get the CXI backend, in the mode that we want. It is possible that
     * some are not necessary, but libfabric does not make it clear which hints are for
     * "turning" things on and which are just for "does this fabric support this".
     * Aside from the obvious hints, FI_RMA_EVENT is needed because we need to track
     * remote writes to specific memory regions.
     *
     * The other unique feature about this function is the binding of the progress counter
     * (LibfabricInstance::progress_ctr). Normally, the binding of the counter to the main
     * endpoint (LibfabricInstance::ep) should only need FI_RECV, but due to a bug we
     * discovered in the CXI implementation, if no counter is set to track FI_WRITE on
     * the endpoint, a segfault happens (the CXI code is missing a null pointer check).
     * Affects 2.3.1 at least, tbd on future versions. For historical purposes, did not
     * affect the ~1.15.2 version that used to be on LLNL systems.
     *
     * @param [in] comm The MPI communicator used to determine the total number of ranks
     * that will be communicated with.
     */
    void initialize_libfabric(MPI_Comm comm);

    /**
     * @brief Collect all of the peer ranks' CXI address so we can talk to them over
     * libfabric.
     * @details One of the perks of building an MPI-based library is that we can use it to
     * help set up our network connections. All ranks get their CXI address, put it into a
     * character array of size 100, and then perform the allgather. CXI addresses are
     * usually quite small (sometimes only four characters), so there should be plenty of
     * space. After the allgather, the addresses are inserted into LibfabricInstance::av
     * in a single go.
     * @param [in] comm The MPI communicator to do the allgather of CXI addresses on.
     */
    void initialize_peer_addresses(MPI_Comm comm);

    /** @brief Helper function for giving memory region ID numbers.
     * @details Libfabric supports the user assigning a key for each memory region, and so
     * this library just decided to do it that way. Note that while the CXI provider does
     * have a concept of "Optimized MRs" for values less than 100, this code does not
     * attempt to explicitly leverage that concept by starting at keys of 1; it starts at
     * 1 just to help make things easier to debug.
     * @return The unique key a new memory region should be assigned.
     */
    static size_t getMRID()
    {
        static size_t ID = 1;
        return ID++;
    }

    /**
     * @brief Private wrapper for allocating a libfabric memory region.
     * @details This function takes care of getting the memory region's key automatically
     * (via LibfabricInstance::getMRID()) as well as binding it to LibfabricInstance::ep.
     * All the relevant options are taken in as parameters.
     * @param [in] buffer The buffer address to associate with the memory region.
     * @param [in] len The size of the buffer, in bytes.
     * @param [in] access Any access flags to pass to fi_mr_reg.
     * @param [in] flags Any other flags to pass to fi_mr_reg.
     * @return A pointer to the newly allocated memory region on the provided buffer.
     */
    struct fid_mr* alloc_mr(const void* buffer, size_t len, uint64_t access,
                            uint64_t flags)
    {
        struct fid_mr* new_mr = nullptr;
        force_libfabric(
            fi_mr_reg(domain, buffer, len, access, 0, getMRID(), flags, &new_mr, NULL));
        force_libfabric(fi_mr_bind(new_mr, &(ep)->fid, 0));
        return new_mr;
    }

    /** @brief Vector of all ranks' CXI addresses. */
    std::vector<fi_addr_t> peers;

    /** @brief Value for keeping track of how many DWQ slots are in use. */
    uint64_t dwq_slots_used = 0;
    /** @brief How many DWQ entries can be in the queue at one time. */
    static constexpr uint64_t MAX_DWQ_SLOTS = 254;
    /** @brief Map of all DWQ progress counters and their last known value.*/
    std::map<fid_cntr*, uint64_t> dwq_progress_counters;
};

/**
 * @brief A wrapper class around a libfabric counter and its CXI extensions
 * @details When using the CXI libfabric provider, it offers various extensions to
 * libfabric that let us access more features of a counter (such as its MMIO address).
 * This class acts as a C++ wrapper around the vendor specific C code, but is otherwise an
 * open container of objects.
 *
 * These counters are not completion counters. These are specifically the counters meant
 * to be triggered by the GPU when it is ready to start a communication.
 *
 * In theory this code should work on NVIDIA systems with slingshot networks, however in
 * practice the step of registering the MMIO address into the GPU memory has failed (as of
 * early 2026), hence the use of HIP in the code.
 */
class CXICounter
{
public:
    /**
     * @brief Create a libfabric counter and use the CXI extensions to map its MMIO
     * address into GPU memory.
     * @details After allocating the counter (which is not tracked for progress), it gets
     * its MMIO address, registers it with the GPU, and finally it gets the GPU device
     * pointer of the counter.
     * @param [in, out] libfab The ::LibfabricInstance containing all of the libfabric
     * objects needed.
     */
    CXICounter(LibfabricInstance& libfab) : counter(libfab.alloc_counter(false))
    {
        // Open (create) CXI Extension object
        force_libfabric(fi_open_ops(&(counter->fid), FI_CXI_COUNTER_OPS, 0,
                                    (void**)&counter_ops, NULL));
        // Get the MMIO Address of the counter
        force_libfabric(
            counter_ops->get_mmio_addr(&counter->fid, &mmio_addr, &mmio_addr_len));
        // Register MMIO Address w/ HIP
        force_gpu(hipHostRegister(mmio_addr, mmio_addr_len, hipHostRegisterDefault));
        // Get GPU version of MMIO address
        force_gpu(hipHostGetDevicePointer(&gpu_mmio_addr, mmio_addr, 0));
    }

    /**
     * @brief Move constructor. Takes everything from @p other and sets it to null values.
     * @param [in, out] other The ::CXICounter to move from.
     */
    CXICounter(CXICounter&& other) noexcept
        : counter(other.counter),
          counter_ops(other.counter_ops),
          mmio_addr(other.mmio_addr),
          mmio_addr_len(other.mmio_addr_len),
          gpu_mmio_addr(other.gpu_mmio_addr),
          use_count(other.use_count)
    {
        other.counter       = nullptr;
        other.counter_ops   = nullptr;
        other.mmio_addr     = nullptr;
        other.mmio_addr_len = 0;
        other.gpu_mmio_addr = nullptr;
        other.use_count     = 0;
    }

    /** @brief Destructor that frees the counter and unregisters the MMIO address, if
     * present.*/
    ~CXICounter()
    {
        if (nullptr != mmio_addr)
        {
            force_gpu(hipHostUnregister(mmio_addr));
        }
        // Free counter
        if (nullptr != counter)
        {
            force_libfabric(fi_close(&counter->fid));
        }
    }

    /** @brief Print the current value of the underlying libfabric counter. */
    void print()
    {
        size_t value = fi_cntr_read(counter);
        Print::out("Value: ", value);
    }

    /** @brief Return what value the counter will need to be at to be triggered.
     * @details Assumes that the caller will call CXICounter::up_use_count() shortly after
     * using this function and submitting a deferred work queue entry. It is essentially
     * the threshold for the current action.
     * @return The value that the counter should be at to trigger the next action.
     */
    size_t get_next_value()
    {
        return use_count + 1;
    }

    /** @brief Increment the counter's threshold (implies that an action has been
     * enqueued).
     * @details Because we are potentially enqueueing up multiple iterations at once, we
     * want to make sure each iteration goes on a specific value. This means we need to
     * use a local variable instead of reading the underlying counter's current value and
     * adding one to that.
     */
    void up_use_count()
    {
        use_count++;
    }

    /** @brief Underlying libfabric counter object that has had its network counter
     * address mapped into GPU memory space.*/
    struct fid_cntr* counter;
    /** @brief CXI Libfabric extension object. */
    struct fi_cxi_cntr_ops* counter_ops;

    /** @brief MMIO address of the libfabric counter. */
    void* mmio_addr;
    /** @brief The size of the MMIO address. */
    size_t mmio_addr_len;
    /** @brief Address of the libfabric counter as it is mapped into GPU memory. */
    void* gpu_mmio_addr;

    /** @brief Keep track of how many triggers have been enqueued. */
    size_t use_count = 0;
};

/**
 * @brief A C++ wrapper class for the libfabric deferred work queue entry
 * @details This class is handy because the main C struct has a union in it that makes
 * typing out the members quite cumbersome. This class is generally a bunch of "set"
 * operations. While it technically provides access to the underlying deferred work queue
 * item via the pointer to it, the user should avoid accessing the struct directly.
 */
class DeferredWorkQueueEntry
{
public:
    /** @brief Create an empty deferred work queue entry with everything set to 0.*/
    DeferredWorkQueueEntry()
    {
        work_entry           = {};
        work_entry.threshold = 0;
    }
    /** @brief Default destructor. */
    ~DeferredWorkQueueEntry() = default;

    /** @brief Copy constructor deleted. */
    DeferredWorkQueueEntry(const DeferredWorkQueueEntry&) = delete;
    /** @brief Copy assignment operator deleted. */
    DeferredWorkQueueEntry& operator=(const DeferredWorkQueueEntry&) = delete;

    /** @brief Move constructor deleted. */
    DeferredWorkQueueEntry(DeferredWorkQueueEntry&& other) = delete;
    /** @brief Move assignment operator deleted. */
    DeferredWorkQueueEntry& operator=(DeferredWorkQueueEntry&& other) = delete;

    /** @brief Gain access to the pointer to the underlying deferred work queue entry.
     * @return The pointer to the deferred work queue entry.
     */
    struct fi_deferred_work* get_dwqe()
    {
        return &work_entry;
    }

    /**
     * @brief Set the libfabric counter that should be used as the completion counter.
     * @details Note that this counter should have been created through
     * LibfabricInstance::alloc_counter() with @p dwq_track set to "true".
     * @param [in] completion_counter The counter object to be used for the completion
     * counter.
     */
    virtual void set_completion_counter(fid_cntr* completion_counter)
    {
        work_entry.completion_cntr = completion_counter;
    }

    /**
     * @brief Set the libfabric counter that should be used as the triggering counter.
     * @param [in] trigger_cntr The CXICounter object that has the underlying libfabric
     * counter that should be used for the triggering counter.
     */
    virtual void set_trigger_counter(CXICounter& trigger_cntr)
    {
        work_entry.triggering_cntr = trigger_cntr.counter;
    }

    /**
     * @brief Set the libfabric counter that should be used as the triggering counter.
     * @details Note that this counter does not need to explicitly be the counter
     * underneath a CXICounter. This is because we can use some counters that track normal
     * libfabric operations (remote writes in this library) and those actions can trigger
     * an operation.
     * @param [in] trigger_counter The counter object to be used for the triggering
     * counter.
     */
    virtual void set_trigger_counter(fid_cntr* trigger_counter)
    {
        work_entry.triggering_cntr = trigger_counter;
    }

    /**
     * @brief Set the threshold that should be used for the deferred work queue entry.
     * @param [in] threshold The value of the threshold to be set.
     */
    virtual void set_threshold(uint64_t threshold)
    {
        work_entry.threshold = threshold;
    }

    /** @brief Manually increase the threshold by one. */
    virtual void bump_threshold()
    {
        work_entry.threshold++;
    }

    /** @brief Prints out the threshold of the deferred work queue entry.*/
    virtual void print()
    {
        Print::always("Threshold:", work_entry.threshold);
    }

protected:
    /** @brief The underlying deferred work queue libfabric object. */
    struct fi_deferred_work work_entry;
};

/**
 * @brief Specialization of ::DeferredWorkQueueEntry that uses the "rma" version of the
 * union.
 * @details Designed for the "rma" operations (fi_write). Like the parent class, it
 * accepts the values for various entries and sets them up in a single call. In addition
 * to the deferred work queue entry that the parent holds, this object also holds onto
 * some of the structs that describe the operation to be performed
 */
class RMAEntry : public DeferredWorkQueueEntry
{
public:
    /**
     * @brief Basic constructor when the peer process is known, and the data descriptions
     * are also known.
     * @details Calls the main constructor first to set up the basic skeleton of the
     * deferred work queue entry and then stores the descriptions about the buffers.
     * @param [in] main_ep The main libfabric endpoint for the CXI fabric.
     * @param [in] partner The peer this operation is going to communicate with.
     * @param [in] send_data An iovec of the data to be moved (sent).
     * @param [in] remote_data The description of the buffer on the remote side where the
     * data should go.
     */
    RMAEntry(struct fid_ep* main_ep, fi_addr_t partner, struct iovec send_data,
             struct fi_rma_iov remote_data)
        : RMAEntry(main_ep, partner)
    {
        msg_iov     = send_data;
        msg_rma_iov = remote_data;
    }

    /**
     * @brief Basic constructor when the peer process is known, but the remote buffer
     * description is not.
     * @details Calls the main constructor first to set up the basic skeleton of the
     * deferred work queue entry and then stores the descriptions about the send buffer
     * only.
     * @param [in] main_ep The main libfabric endpoint for the CXI fabric.
     * @param [in] partner The peer this operation is going to communicate with.
     * @param [in] send_data An iovec of the data to be moved (sent).
     */
    RMAEntry(struct fid_ep* main_ep, fi_addr_t partner, struct iovec send_data)
        : RMAEntry(main_ep, partner)
    {
        // Update local iovec to what needs to be sent
        msg_iov = send_data;
    }

    /**
     * @brief Main constructor. Assumes only peer is known, and nothing about the buffers
     * is known.
     * @details Currently only supports "FI_OP_WRITE" operations.
     * @param [in] main_ep The main libfabric endpoint for the CXI fabric.
     * @param [in] partner The peer this operation is going to communicate with.
     */
    RMAEntry(struct fid_ep* main_ep, fi_addr_t partner) : DeferredWorkQueueEntry()
    {
        // Adjustments to base DWQ entry because of our type
        work_entry.op_type = FI_OP_WRITE;
        work_entry.op.rma  = &rma_work;
        // Setting up our RMA op
        rma_work.ep    = main_ep;
        rma_work.flags = 0;
        // Setting up the send buffer iov info (NO ACTUAL BUFFER DATA)
        msg_iov                = {0, 0};
        rma_work.msg.msg_iov   = &msg_iov;
        rma_work.msg.iov_count = 1;
        // To who are we going to
        rma_work.msg.addr = partner;
        // Setting up remote iov info (NO ACTUAL BUFFER DATA)
        msg_rma_iov                = {0, 0, 0};
        rma_work.msg.rma_iov       = &msg_rma_iov;
        rma_work.msg.rma_iov_count = 1;
    }

    /** @brief Get the pointer to the underlying remote buffer details.
     * @return The pointer to the underlying remote buffer details.
     */
    struct fi_rma_iov* get_rma_iov_addr()
    {
        return &msg_rma_iov;
    }

    /**
     * @brief Update the details about the buffer to send data from.
     * @param [in] new_iovec Details about the buffer location and the size of the data to
     * be moved.
     */
    void set_iovec(iovec new_iovec)
    {
        msg_iov = new_iovec;
    }

    /**
     * @brief Update the flags for the rma operation.
     * @details The constructor sets no flags on the operation.
     * @param [in] flags The new flags to set
     */
    void set_flags(uint64_t flags)
    {
        rma_work.flags = flags;
    }

    /**
     * @brief Update the details about the remote buffer.
     * @param [in] new_iov The new details about the remote buffer.
     */
    void set_rma_iov(fi_rma_iov new_iov)
    {
        msg_rma_iov = new_iov;
    }

    /** @brief Print out details about the remote and local buffers. */
    void print() override
    {
        DeferredWorkQueueEntry::print();
        Print::always("Local IOVEC:", msg_iov.iov_base, msg_iov.iov_len);
        Print::always("Remote IOVEC:", msg_rma_iov.addr, msg_rma_iov.len,
                      msg_rma_iov.key);
    }

protected:
    /** @brief An overarching struct that contains details about the RMA operation. */
    struct fi_op_rma rma_work;
    /** @brief Details about the send buffer (the buffer to move data from). */
    struct iovec msg_iov;
    /** @brief Details about the remote buffer. */
    struct fi_rma_iov msg_rma_iov;
};

/**
 * @brief Specialization of ::DeferredWorkQueueEntry that uses the "atomic" version of the
 * union.
 * @details Designed for the "atomic" operations (fi_atomic). Like the parent class, it
 * accepts the values for various entries and sets them up in a single call. In addition
 * to the deferred work queue entry that the parent holds, this object also holds onto
 * some of the structs that describe the operation to be performed.
 */
class AtomicEntry : public DeferredWorkQueueEntry
{
public:
    /**
     * @brief Main constructor. Assumes only peer is known, and nothing about the buffers
     * is known.
     * @details This constructor assumes that the operation is "FI_SUM" and is using 1
     * "FI_UINT64".
     * @param [in] main_ep The main libfabric endpoint for the CXI fabric.
     * @param [in] partner The peer this operation is going to communicate with.
     */
    AtomicEntry(struct fid_ep* main_ep, fi_addr_t partner) : DeferredWorkQueueEntry()
    {
        work_entry.op_type   = FI_OP_ATOMIC;
        work_entry.op.atomic = &atomic_work;

        atomic_work       = {};
        atomic_work.flags = 0;
        atomic_work.ep    = main_ep;

        msg_ioc                   = {&buffer_value, 1};
        atomic_work.msg.msg_iov   = &msg_ioc;
        atomic_work.msg.iov_count = 1;
        atomic_work.msg.addr      = partner;
        atomic_work.msg.datatype  = FI_UINT64;
        atomic_work.msg.op        = FI_SUM;

        msg_rma_ioc                   = {0, 0, 0};
        atomic_work.msg.rma_iov       = &msg_rma_ioc;
        atomic_work.msg.rma_iov_count = 1;
    }

    /**
     * @brief Basic constructor when the peer process and the remote buffer
     * description are known.
     * @param [in] main_ep The main libfabric endpoint for the CXI fabric.
     * @param [in] partner The peer this operation is going to communicate with.
     * @param [in] remote_data The description of the buffer on the remote side where the
     * data should go.
     */
    AtomicEntry(struct fid_ep* main_ep, fi_addr_t partner, struct fi_rma_ioc remote_data)
        : AtomicEntry(main_ep, partner)
    {
        msg_rma_ioc = remote_data;
    }

    /** @brief Get the pointer to the underlying remote buffer's details.
     * @return The pointer to the struct containing the remote buffer's details.
     */
    struct fi_rma_ioc* get_rma_ioc_addr()
    {
        return (&msg_rma_ioc);
    }

    /**
     * @brief Set details about the remote buffer.
     * @param [in] new_rma_ioc The new details about the remote buffer.
     */
    void set_rma_ioc(struct fi_rma_ioc new_rma_ioc)
    {
        msg_rma_ioc = new_rma_ioc;
    }

    /**
     * @brief Update the flags for the atomic operation.
     * @details The constructor sets no flags on the operation.
     * @param [in] flags The new flags to set
     */
    void set_flags(uint64_t flags)
    {
        atomic_work.flags = flags;
    }

    /** @brief Print out details about the remote and local buffers.*/
    void print() override
    {
        DeferredWorkQueueEntry::print();
        Print::always("Local IOC:", msg_ioc.addr, msg_ioc.count);
        Print::always("Remote IOC:", msg_rma_ioc.addr, msg_rma_ioc.count,
                      msg_rma_ioc.key);
    }

protected:
    /** @brief The "buffer" that is used as the value to compute with during the atomic
     * operation.
     * @details Currently this library only wants to do atomic adds of a value of "1",
     * which is why this value is set to 1.
     */
    uint64_t buffer_value = 1;
    /** @brief An overarching struct that contains details about the atomic operation. */
    struct fi_op_atomic atomic_work;
    /** @brief Details about the buffer to contribute to the atomic.
     * @details Currently just set to AtomicEntry::buffer_value.
     */
    struct fi_ioc msg_ioc;
    /** @brief Details about the remote buffer. */
    struct fi_rma_ioc msg_rma_ioc;
};

#endif