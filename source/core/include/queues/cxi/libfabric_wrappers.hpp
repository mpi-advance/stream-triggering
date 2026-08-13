#ifndef ST_CXI_QUEUE_LIBFABRIC_WRAPPER
#define ST_CXI_QUEUE_LIBFABRIC_WRAPPER

#include <map>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_trigger.h>
// clang-format off
#include <rdma/fi_cxi_ext.h>
// clang-format on

#include "misc/print.hpp"
#include "safety/libfabric.hpp"
#include "safety/mpi.hpp"

class LibfabricInstance
{
public:
    LibfabricInstance() = default;
    ~LibfabricInstance();

    void initialize(MPI_Comm comm)
    {
        comm_size = -1;
        check_mpi(MPI_Comm_size(comm, &comm_size));
        initialize_libfabric();
        initialize_peer_addresses(comm);
    }

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

    struct fid_mr* create_mr(const void* buffer, size_t len, uint64_t access,
                             uint64_t flags)
    {
        // Enable MR
        struct fid_mr* new_mr = alloc_mr(buffer, len, access, flags);
        force_libfabric(fi_mr_enable(new_mr));
        Print::out("Buffer:", buffer, "has mr key: ", fi_mr_key(new_mr));
        return new_mr;
    }

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

    fi_addr_t get_peer(int rank)
    {
        return peers.at(rank);
    }

    void queue_work(struct fi_deferred_work* work_entry)
    {
        Print::out("<H> Threshold:", work_entry->threshold, work_entry->triggering_cntr,
                   fi_cntr_read(work_entry->triggering_cntr), work_entry->completion_cntr,
                   fi_cntr_read(work_entry->completion_cntr));

        while (dwq_slots_used == MAX_DWQ_SLOTS)
        {
            progress_dwq();
        }
        force_libfabric(fi_control(&domain->fid, FI_QUEUE_WORK, work_entry));
        dwq_slots_used++;
    }

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

    struct fi_info*    fi;           /*!< Provider's data and features */
    struct fid_fabric* fabric;       /*!< Represents the network */
    struct fid_domain* domain;       /*!< A subsection of the network */
    struct fid_av*     av;           /*!< Address vector for connections */
    struct fid_ep*     ep;           /*!< An endpoint */
    struct fid_cq*     txcq;         /*!< The transmit completion queue */
    struct fid_cq*     rxcq;         /*!< The receive completion queue */
    struct fid_cntr*   progress_ctr; /*!< The counters for receiving */

private:
    void select_fi_nic(fi_info*&);
    void initialize_libfabric();
    void initialize_peer_addresses(MPI_Comm comm);

    // MR Management
    static size_t getMRID()
    {
        static size_t ID = 1;
        return ID++;
    }

    struct fid_mr* alloc_mr(const void* buffer, size_t len, uint64_t access,
                            uint64_t flags)
    {
        struct fid_mr* new_mr = nullptr;
        force_libfabric(
            fi_mr_reg(domain, buffer, len, access, 0, getMRID(), flags, &new_mr, NULL));
        force_libfabric(fi_mr_bind(new_mr, &(ep)->fid, 0));
        return new_mr;
    }

    int                    comm_size;
    std::vector<fi_addr_t> peers;

    uint64_t                      dwq_slots_used = 0;
    uint64_t                      MAX_DWQ_SLOTS  = 254;
    std::map<fid_cntr*, uint64_t> dwq_progress_counters;
};

class CXICounter
{
public:
    CXICounter(LibfabricInstance& libfab) : counter(libfab.alloc_counter(false))
    {
        // Open (create) CXI Extension object
        check_libfabric(fi_open_ops(&(counter->fid), FI_CXI_COUNTER_OPS, 0,
                                    (void**)&counter_ops, NULL));
        // Get the MMIO Address of the counter
        check_libfabric(
            counter_ops->get_mmio_addr(&counter->fid, &mmio_addr, &mmio_addr_len));
        // Register MMIO Address w/ HIP
        force_gpu(hipHostRegister(mmio_addr, mmio_addr_len, hipHostRegisterDefault));
        // Get GPU version of MMIO address
        force_gpu(hipHostGetDevicePointer(&gpu_mmio_addr, mmio_addr, 0));
    }

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

    ~CXICounter()
    {
        // Free counter
        if (nullptr != counter)
        {
            force_libfabric(fi_close(&counter->fid));
        }
        if (nullptr != mmio_addr)
        {
            force_gpu(hipHostUnregister(mmio_addr));
        }
    }

    void print()
    {
        size_t value = fi_cntr_read(counter);
        Print::out("Value: ", value);
    }

    size_t get_next_value()
    {
        return use_count + 1;
    }

    void up_use_count()
    {
        use_count++;
    }

    // Libfabric Structs
    struct fid_cntr*        counter;
    struct fi_cxi_cntr_ops* counter_ops;

    // MMIO Pointers
    void*  mmio_addr;
    size_t mmio_addr_len;
    void*  gpu_mmio_addr;

    // Keep track of how many times it was triggered
    size_t use_count = 0;
};


class DeferredWorkQueueEntry
{
public:
    DeferredWorkQueueEntry()
    {
        work_entry           = {};
        work_entry.threshold = 0;
    }

    struct fi_deferred_work* get_dwqe()
    {
        return &work_entry;
    }

    virtual void set_completion_counter(fid_cntr* completion_counter)
    {
        work_entry.completion_cntr = completion_counter;
    }

    virtual void set_trigger_counter(CXICounter& trigger_cntr)
    {
        work_entry.triggering_cntr = trigger_cntr.counter;
    }

    virtual void set_trigger_counter(fid_cntr* trigger_counter)
    {
        work_entry.triggering_cntr = trigger_counter;
    }

    virtual void set_threshold(uint64_t threshold)
    {
        work_entry.threshold = threshold;
    }

    virtual void bump_threshold()
    {
        work_entry.threshold++;
    }

    virtual void print()
    {
        Print::always("Threshold:", work_entry.threshold);
    }

protected:
    struct fi_deferred_work work_entry;
};

class RMAEntry : public DeferredWorkQueueEntry
{
public:
    // Know what to send and where it's going
    RMAEntry(struct fid_ep* main_ep, fi_addr_t partner, struct iovec send_data,
             struct fi_rma_iov remote_data)
        : RMAEntry(main_ep, partner)
    {
        msg_iov     = send_data;
        msg_rma_iov = remote_data;
    }

    // Know what to send, but don't know where data is going
    RMAEntry(struct fid_ep* main_ep, struct iovec send_data, fi_addr_t partner)
        : RMAEntry(main_ep, partner)
    {
        // Update local iovec to what needs to be sent
        msg_iov = send_data;
    }

    // Don't know what to send and don't know where data is going
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

    struct fi_rma_iov* get_rma_iov_addr()
    {
        return &msg_rma_iov;
    }

    void set_iovec(iovec new_iovec)
    {
        msg_iov = new_iovec;
    }

    void set_flags(uint64_t flags)
    {
        rma_work.flags = flags;
    }

    void set_rma_iov(fi_rma_iov new_iov)
    {
        msg_rma_iov = new_iov;
    }

    void print() override
    {
        DeferredWorkQueueEntry::print();
        Print::always("Local IOVEC:", msg_iov.iov_base, msg_iov.iov_len);
        Print::always("Remote IOVEC:", msg_rma_iov.addr, msg_rma_iov.len,
                      msg_rma_iov.key);
    }

protected:
    struct fi_op_rma  rma_work;
    struct iovec      msg_iov;
    struct fi_rma_iov msg_rma_iov;
};

class AtomicEntry : public DeferredWorkQueueEntry
{
public:
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

    AtomicEntry(struct fid_ep* main_ep, fi_addr_t partner, struct fi_rma_ioc* remote_data)
        : AtomicEntry(main_ep, partner)
    {
        msg_rma_ioc = {remote_data->addr, remote_data->count, remote_data->key};
    }

    struct fi_rma_ioc* get_rma_ioc_addr()
    {
        return (&msg_rma_ioc);
    }

    void set_rma_ioc(struct fi_rma_ioc new_rma_ioc)
    {
        msg_rma_ioc = new_rma_ioc;
    }

    void set_flags(uint64_t flags)
    {
        atomic_work.flags = flags;
    }

    void print() override
    {
        DeferredWorkQueueEntry::print();
        Print::always("Local IOC:", msg_ioc.addr, msg_ioc.count);
        Print::always("Remote IOC:", msg_rma_ioc.addr, msg_rma_ioc.count,
                      msg_rma_ioc.key);
    }

protected:
    size_t              buffer_value = 1;
    struct fi_op_atomic atomic_work;
    struct fi_ioc       msg_ioc;
    struct fi_rma_ioc   msg_rma_ioc;
};

#endif