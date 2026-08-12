
#ifndef ST_CXI_QUEUE
#define ST_CXI_QUEUE

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_trigger.h>
// clang-format off
#include <rdma/fi_cxi_ext.h>
// clang-format on

#include <map>
#include <numeric>
#include <vector>

#include "abstract/match.hpp"
#include "misc/print.hpp"
#include "queues/HIPQueue.hpp"
#include "safety/gpu.hpp"
#include "safety/libfabric.hpp"
#include "safety/mpi.hpp"

static inline size_t get_size_of_buffer(Request& req)
{
    int size = -1;
    check_mpi(MPI_Type_size(req.datatype, &size));
    return (size_t)(size * req.count);
}

class CompletionBuffer
{
public:
    CompletionBuffer(void* _address, size_t _len, size_t _count, uint64_t _mr_key,
                     uint64_t _offset)
        : address(_address),
          iov_description({_offset, _len, _mr_key}),
          ioc_description({_offset, _count, _mr_key})
    {
    }

    CompletionBuffer()
        : address(nullptr), iov_description({0, 0, 0}), ioc_description({0, 0, 0})
    {
    }

    operator fi_rma_iov() const
    {
        return iov_description;
    }

    operator fi_rma_ioc() const
    {
        return ioc_description;
    }

    struct fi_rma_iov* get_rma_iov_addr()
    {
        return &iov_description;
    }

    struct fi_rma_ioc* get_rma_ioc_addr()
    {
        return &ioc_description;
    }

    void print()
    {
        Print::out("CB:", address, iov_description.addr, iov_description.len,
                   iov_description.key, "/", ioc_description.addr, ioc_description.count,
                   ioc_description.key);
    }

    void*             address;
    struct fi_rma_iov iov_description;
    struct fi_rma_ioc ioc_description;
};

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

class CompletionBufferFactory
{
public:
    CompletionBufferFactory() : my_mr(nullptr), current_index(0)
    {
        initialize_main_buffer();
    }

    ~CompletionBufferFactory()
    {
        if (my_mr)
        {
            force_libfabric(fi_close(&(my_mr)->fid));
        }
        check_gpu(hipFree(buffer));
    }

    CompletionBufferFactory(const CompletionBufferFactory& other) = delete;
    CompletionBufferFactory(CompletionBufferFactory&& other)
    {
        buffer       = other.buffer;
        my_mr        = other.my_mr;
        other.buffer = nullptr;
        other.my_mr  = nullptr;
    }

    CompletionBufferFactory& operator=(const CompletionBufferFactory& rhs) = delete;
    CompletionBufferFactory& operator=(CompletionBufferFactory&& other)
    {
        buffer       = other.buffer;
        my_mr        = other.my_mr;
        other.buffer = nullptr;
        other.my_mr  = nullptr;
        return *this;
    }

    hipIpcMemHandle_t get_ipc_handle()
    {
        return ipc_handle;
    }

    void register_mr(LibfabricInstance& libfab)
    {
        my_mr = libfab.create_mr(buffer, DEFAULT_SIZE, FI_REMOTE_WRITE | FI_WRITE,
                                 FI_MR_ALLOCATED);
    }

    void free_mr()
    {
        check_libfabric(fi_close(&(my_mr)->fid));
        my_mr = nullptr;
    }

    CompletionBuffer alloc_buffer()
    {
        if (current_index >= DEFAULT_ITEMS)
            throw std::runtime_error("Out of space for completion buffer");
        if (nullptr == my_mr)
            throw std::runtime_error("Buffer is not registered with libfabric");
        void*    x            = ((char*)buffer) + (sizeof(size_t) * current_index);
        uint64_t offset_value = current_index * DEFAULT_ITEM_SIZE;
        current_index++;
        return CompletionBuffer(x, DEFAULT_ITEM_SIZE, 1, fi_mr_key(my_mr), offset_value);
    }

    using COMPLETION_TYPE                       = size_t;
    static constexpr size_t DEFAULT_ITEMS       = 1000;
    static constexpr size_t DEFAULT_ITEM_SIZE   = sizeof(COMPLETION_TYPE);
    static constexpr size_t DEFAULT_SIZE        = DEFAULT_ITEMS * DEFAULT_ITEM_SIZE;
    static constexpr size_t MAX_GPU_COMPLETIONS = 100;

private:
    struct fid_mr*    my_mr;
    void*             buffer;
    size_t            current_index;
    hipIpcMemHandle_t ipc_handle;

    void initialize_main_buffer()
    {
        force_gpu(hipMalloc(&buffer, DEFAULT_SIZE));
        force_gpu(hipMemset(buffer, 0, DEFAULT_SIZE));
        force_gpu(hipDeviceSynchronize());
        force_gpu(hipIpcGetMemHandle(&ipc_handle, buffer));
        Print::out("Default Completion Buffer location:", buffer);
    }
};

struct GPUCompletionDescription
{
    volatile CompletionBufferFactory::COMPLETION_TYPE* comp_addr;
    size_t                                             goal_value;
};

struct GPUCompletions
{
    GPUCompletionDescription buffer[CompletionBufferFactory::MAX_GPU_COMPLETIONS];
    size_t                   actual_size;
};

using GPUTriggerDescription = uint64_t*;
struct GPUTriggers
{
    GPUTriggerDescription buffer[CompletionBufferFactory::MAX_GPU_COMPLETIONS];
    size_t                actual_size;
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

enum class TriggerStatus
{
    NOT_NEEDED  = 0,
    GLOBAL_BUMP = 1,
    EXTRA_BUMP  = 2,
    DONE        = 3,  // No wait
};

enum class WaitStatus
{
    NOT_NEEDED    = 0,
    GLOBAL_KERNEL = 1,
};

class CXIRequest
{
public:
    CXIRequest(Request& req, CompletionBufferFactory& buffers)
        : base_req(req), completion_buffer(buffers.alloc_buffer()), num_times_started(0)
    {
    }

    // Delayed (or no) buffer setup
    CXIRequest(Request& req) : base_req(req), num_times_started(0) {}

    virtual ~CXIRequest() = default;
    virtual TriggerStatus start_cpu(CXICounter& trigger_cntr, hipStream_t* the_stream)
    {
        num_times_started++;
        return start_derived(trigger_cntr, the_stream);
    }

    virtual WaitStatus wait_gpu(hipStream_t* the_stream);

    virtual GPUMemoryType get_gpu_memory_type()
    {
        return base_req.get_memory_type();
    }

    virtual void match(MPI_Comm phase_a, MPI_Comm phase_b) = 0;

    GPUCompletionDescription get_gpu_completion()
    {
        return {(size_t*)completion_buffer.address, num_times_started};
    }

    virtual GPUTriggerDescription get_gpu_trigger()
    {
        throw std::runtime_error("Request was unable to offer trigger description");
    }

protected:
    virtual TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                        hipStream_t* the_stream) = 0;

    Request&         base_req;
    CompletionBuffer completion_buffer;
    CompletionBuffer protocol_buffer;

    size_t num_times_started;
};

class CXIRSend : public CXIRequest
{
public:
    CXIRSend(Request& user_request, CompletionBufferFactory& buffers,
             LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRequest(user_request, buffers),
          work_entry(_libfab.ep,
                     {user_request.send_buffer,
                      static_cast<size_t>(get_size_of_buffer(user_request))},
                     _libfab.get_peer(user_request.resolve_comm_world())),
          local_completion(_libfab.ep, self, completion_buffer.get_rma_ioc_addr()),
          libfab(_libfab),
          completion_a(_libfab.alloc_counter(true)),
          completion_b(_libfab.alloc_counter(true))
    {
        work_entry.set_completion_counter(completion_a);
        local_completion.set_trigger_counter(completion_a);
        local_completion.set_completion_counter(completion_b);
    }

    ~CXIRSend()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_b);
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::sender_rndv(work_entry.get_rma_iov_addr(),
                                                  protocol_buffer.get_rma_ioc_addr(),
                                                  base_req, comm_a, comm_b);
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        // Update threshold of chained things
        work_entry.set_threshold(trigger_cntr.get_next_value());
        // Adjust the triggering counter to use
        work_entry.set_trigger_counter(trigger_cntr);

        // Queue up send of user data
        // work_entry.print();
        libfab.queue_work(work_entry.get_dwqe());

        // Queue up completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        return TriggerStatus::GLOBAL_BUMP;
    }

protected:
    // Structs for the DFWQ Entry:
    RMAEntry    work_entry;
    AtomicEntry local_completion;

    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    struct fid_cntr* completion_a;
    struct fid_cntr* completion_b;
};

class CXISend : public CXIRSend
{
public:
    CXISend(Request& user_request, CompletionBufferFactory& buffers,
            LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRSend(user_request, buffers, _libfab, self), triggered(_libfab)
    {
        work_entry.set_trigger_counter(triggered);

        /* Setup CTS buffer */
        force_gpu(hipHostMalloc(&cts_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE,
                                hipHostMallocDefault));
        cts_mr = _libfab.create_mr_with_counter(
            cts_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE, FI_REMOTE_WRITE,
            FI_MR_ALLOCATED | FI_RMA_EVENT, triggered.counter, FI_REMOTE_WRITE);

        /* Set Protocol Buffer */
        protocol_buffer =
            CompletionBuffer(cts_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE, 1,
                             fi_mr_key(cts_mr), 0);
    }

    ~CXISend()
    {
        /* Free CTS related items */
        check_gpu(hipHostFree(cts_buffer));
        check_libfabric(fi_close(&(cts_mr)->fid));
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        // Update threshold of chained things
        work_entry.set_threshold(triggered.get_next_value() * 2);

        // Queue up send of user data
        // work_entry.print();
        libfab.queue_work(work_entry.get_dwqe());

        // Queue up completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        // triggered.enqueue_trigger(the_stream);
        return TriggerStatus::EXTRA_BUMP;
    }

    GPUTriggerDescription get_gpu_trigger() override
    {
        triggered.up_use_count();
        return (uint64_t*)(triggered.gpu_mmio_addr);
    }

private:
    CXICounter triggered;
    void*      cts_buffer;
    fid_mr*    cts_mr;
};

class CXIRSendShared : public CXIRequest
{
public:
    CXIRSendShared(Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers),
          peer_buffer_ptr(nullptr),
          peer_completion_ptr(nullptr),
          first_time(true)
    {
    }

    ~CXIRSendShared()
    {
        if (!first_time)
        {
            check_gpu(hipIpcCloseMemHandle(peer_buffer_ptr));
            check_gpu(hipIpcCloseMemHandle(peer_completion_ptr));
        }
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::sender_hip_ipc(ipc_data, base_req, comm_a);
    }

protected:
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        if (first_time)
        {
            Print::out(base_req.getID(), "Opening handles for send request!");
            force_gpu(hipIpcOpenMemHandle(&peer_buffer_ptr, ipc_data[0].handle,
                                          hipIpcMemLazyEnablePeerAccess));
            force_gpu(hipIpcOpenMemHandle(&peer_completion_ptr, ipc_data[1].handle,
                                          hipIpcMemLazyEnablePeerAccess));
            first_time = false;
        }
        Print::out("Using Offsets for IPC:", ipc_data[0].offset, ipc_data[1].offset,
                   get_size_of_buffer(base_req), peer_buffer_ptr);

        void* peer_true_buffer = (char*)peer_buffer_ptr + ipc_data[0].offset;
        force_gpu(hipMemcpyDtoDAsync(peer_true_buffer, base_req.send_buffer,
                                     get_size_of_buffer(base_req), *the_stream));

        void* peer_true_completion = (char*)peer_completion_ptr + ipc_data[1].offset;
        force_gpu(hipMemcpyDtoDAsync(peer_true_completion, &num_times_started,
                                     sizeof(num_times_started), *the_stream));

        return TriggerStatus::DONE;
    }

private:
    void* peer_buffer_ptr;
    void* peer_completion_ptr;
    bool  first_time;
    // [0] = remote buffer, [1] = remote completion
    std::array<ProtocolMatch::IPCBundle, 2> ipc_data;
};

class CXIRSendSelf : public CXIRequest
{
public:
    CXIRSendSelf(Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::sender_self(remote_data, base_req, comm_a, comm_b);
    }

protected:
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        Print::out("Self send wants to go to:", remote_data[0], remote_data[1]);

        force_gpu(hipMemcpyDtoDAsync(remote_data[0], base_req.send_buffer,
                                     get_size_of_buffer(base_req), *the_stream));

        force_gpu(hipMemcpyDtoDAsync(remote_data[1], &num_times_started,
                                     sizeof(num_times_started), *the_stream));

        return TriggerStatus::DONE;
    }

private:
    // [0] = remote buffer, [1] = remote completion
    Communication::ProtocolMatch::SelfBundle remote_data;
};

class CXISendCredit : public CXIRSend
{
public:
    CXISendCredit(Request& user_request, CompletionBufferFactory& buffers,
                  LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRSend(user_request, buffers, _libfab, self)
    {
        credit_slots.reserve(Communication::MAX_CREDIT_SLACK);
        for (size_t index = 0; index < Communication::MAX_CREDIT_SLACK; index++)
        {
            CXICounter temp_credit_counter(_libfab);
            void*      temp_credit_buffer;
            force_gpu(hipHostMalloc(&temp_credit_buffer,
                                    CompletionBufferFactory::DEFAULT_ITEM_SIZE,
                                    hipHostMallocDefault));
            fid_mr* temp_credit_mr = _libfab.create_mr_with_counter(
                temp_credit_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE,
                FI_REMOTE_WRITE, FI_MR_ALLOCATED | FI_RMA_EVENT,
                temp_credit_counter.counter, FI_REMOTE_WRITE);
            credit_slots.emplace_back(std::move(temp_credit_counter), temp_credit_buffer,
                                      temp_credit_mr);
            Print::out("CREDIT MR SLOT:", temp_credit_buffer,
                       temp_credit_counter.gpu_mmio_addr, temp_credit_mr);
            credit_buffers[index] = {0, 1, fi_mr_key(temp_credit_mr)};
        }
    }

    ~CXISendCredit()
    {
        for (auto& slot : credit_slots)
        {
            force_gpu(hipHostFree(std::get<1>(slot)));
            check_libfabric(fi_close(&(std::get<2>(slot))->fid));
        }
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::sender_credit(
            &credit_buffer_details, credit_buffers, base_req, comm_a, comm_b);
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        size_t mr_counter_index =
            (num_times_started - 1) % Communication::MAX_CREDIT_SLACK;
        size_t threshold =
            ((num_times_started - 1) / Communication::MAX_CREDIT_SLACK) * 2 + 1;

        /* Update work entry to the correct remote buffer data */
        auto     user_buffer_size  = get_size_of_buffer(base_req);
        uint64_t offset            = mr_counter_index * user_buffer_size;
        credit_buffer_details.addr = offset;
        credit_buffer_details.len  = user_buffer_size;
        Print::out("Using:", credit_buffer_details.addr, credit_buffer_details.len,
                   credit_buffer_details.key);
        work_entry.set_rma_iov(credit_buffer_details);
        /* Update work entry to use current credit threshold */
        work_entry.set_threshold(threshold);
        /* Adjust the triggering counter to use the credit counter */
        work_entry.set_trigger_counter(std::get<0>(credit_slots.at(mr_counter_index)));

        // Queue up send of user data
        // work_entry.print();
        libfab.queue_work(work_entry.get_dwqe());

        // Queue up completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        return TriggerStatus::EXTRA_BUMP;
    }

    GPUTriggerDescription get_gpu_trigger() override
    {
        size_t mr_counter_index =
            (num_times_started - 1) % Communication::MAX_CREDIT_SLACK;
        auto& triggered = std::get<0>(credit_slots.at(mr_counter_index));
        triggered.up_use_count();
        return (uint64_t*)(triggered.gpu_mmio_addr);
    }

private:
    std::vector<std::tuple<CXICounter, void*, fid_mr*>> credit_slots;
    struct fi_rma_iov                                   credit_buffer_details;
    std::array<ProtocolMatch::CreditBundle, Communication::MAX_CREDIT_SLACK>
        credit_buffers;
};

template <bool USE_EAGER>
class CXIRecvOneSided : public CXIRequest
{
public:
    CXIRecvOneSided(Request& user_request, CompletionBufferFactory& buffers,
                    LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRequest(user_request, buffers),
          libfab(_libfab),
          cts_entry(_libfab.ep, _libfab.get_peer(user_request.resolve_comm_world())),
          completion_a(_libfab.alloc_counter(true)),   // CTS DWQ completion tracker
          completion_b(_libfab.alloc_counter(false)),  // registered with user MR
          completion_c(_libfab.alloc_counter(true)),   // Local completion DWQ Tracker
          local_completion(_libfab.ep, self, completion_buffer.get_rma_ioc_addr())
    {
        my_mr = _libfab.create_mr_with_counter(
            user_request.recv_buffer, get_size_of_buffer(user_request), FI_REMOTE_WRITE,
            FI_MR_ALLOCATED | FI_RMA_EVENT, completion_b, FI_REMOTE_WRITE);

        user_buffer_rma_iov = {0, get_size_of_buffer(user_request), fi_mr_key(my_mr)};

        cts_entry.set_completion_counter(completion_a);
        local_completion.set_trigger_counter(completion_b);
        local_completion.set_completion_counter(completion_c);
    }

    ~CXIRecvOneSided()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_c);
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
        libfab.dealloc_counter(completion_b);
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::receiver_rndv(
            &user_buffer_rma_iov, cts_entry.get_rma_ioc_addr(), base_req, comm_a, comm_b);
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        TriggerStatus rc = TriggerStatus::NOT_NEEDED;
        if constexpr (!USE_EAGER)
        {
            Print::out("Queue CTS to Libfabric!");
            // Update threshold of chained things
            cts_entry.set_threshold(trigger_cntr.get_next_value());
            // Adjust the triggering counter to use
            cts_entry.set_trigger_counter(trigger_cntr);
            libfab.queue_work(cts_entry.get_dwqe());
            rc = TriggerStatus::GLOBAL_BUMP;
        }

        // Queue up completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        return rc;
    }

private:
    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    // CTS Preparations
    struct fid_cntr* completion_a;
    AtomicEntry      cts_entry;

    // Allocated Libfabric Objects
    struct fid_mr* my_mr;

    // User buffer details
    struct fi_rma_iov user_buffer_rma_iov;

    // Local completion
    struct fid_cntr* completion_b;
    struct fid_cntr* completion_c;
    AtomicEntry      local_completion;
};

class CXIRecvShared : public CXIRequest
{
public:
    CXIRecvShared(Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
        /* Fill in ipc_data with blanks for user data, but real completion buffer IPC
         * handle and offset */
        ipc_data = {
            {{0, 0}, {buffers.get_ipc_handle(), completion_buffer.iov_description.addr}}};
        /* Get the IPC handle of user's buffer */
        force_gpu(hipIpcGetMemHandle(&(ipc_data[0].handle), user_request.recv_buffer));
        /* Figure out user buffers' original allocation, and offset into it. */
        void*  original_ptr;
        size_t original_size;
        force_gpu(hipMemGetAddressRange(&original_ptr, &original_size,
                                        user_request.recv_buffer));
        ipc_data[0].offset = (char*)user_request.recv_buffer - (char*)original_ptr;
        Print::out("IPC User Offset:", ipc_data[0].offset,
                   "IPC Comp Offset:", ipc_data[1].offset);
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::receiver_hip_ipc(ipc_data, base_req, comm_a);
    }

protected:
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        return TriggerStatus::NOT_NEEDED;
    }

private:
    // [0] = buffer, [1] = completion
    std::array<ProtocolMatch::IPCBundle, 2> ipc_data;
};

class CXIRecvSelf : public CXIRequest
{
public:
    CXIRecvSelf(Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
        self_data[0] = user_request.recv_buffer;
        self_data[1] = completion_buffer.address;
        Print::out("Self wants to go to:", self_data[0], self_data[1]);
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        ProtocolMatch::receiver_self(self_data, base_req, comm_a, comm_b);
    }

protected:
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        return TriggerStatus::NOT_NEEDED;
    }

private:
    // [0] = buffer, [1] = completion
    ProtocolMatch::SelfBundle self_data;
};

class CXIRecvCredit : public CXIRequest
{
public:
    CXIRecvCredit(Request& user_request, CompletionBufferFactory& buffers,
                  LibfabricInstance& _libfab, fi_addr_t self)

        : CXIRequest(user_request, buffers),
          libfab(_libfab),
          triggered(_libfab),
          credit_entry(_libfab.ep, _libfab.get_peer(user_request.resolve_comm_world())),
          completion_a(_libfab.alloc_counter(true)),   // CTS DWQ completion tracker
          completion_b(_libfab.alloc_counter(false)),  // registered with user MR
          completion_c(_libfab.alloc_counter(true)),   // Local completion DWQ Tracker
          local_completion(_libfab.ep, self, completion_buffer.get_rma_ioc_addr())
    {
        auto buffer_size =
            get_size_of_buffer(user_request) * Communication::MAX_CREDIT_SLACK;
        force_gpu(hipMalloc(&credit_buffers, buffer_size));
        my_mr = _libfab.create_mr_with_counter(
            credit_buffers, buffer_size, FI_REMOTE_WRITE, FI_MR_ALLOCATED | FI_RMA_EVENT,
            completion_b, FI_REMOTE_WRITE);

        credit_buffer_rma_iov = {0, buffer_size, fi_mr_key(my_mr)};

        // Adjust the triggering counter to use
        credit_entry.set_trigger_counter(triggered);
        credit_entry.set_completion_counter(completion_a);
        local_completion.set_trigger_counter(completion_b);
        local_completion.set_completion_counter(completion_c);
    }

    ~CXIRecvCredit()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_c);
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
        libfab.dealloc_counter(completion_b);
        check_gpu(hipFree(credit_buffers));
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::receiver_credit(
            &credit_buffer_rma_iov, remote_credit_buffers, base_req, comm_a, comm_b);
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        Print::out("Queue Credit to Libfabric!");
        // Update which credit slot we want to write to
        credit_entry.set_rma_ioc(remote_credit_buffers.at(
            ((num_times_started - 1) % Communication::MAX_CREDIT_SLACK)));
        // Update threshold of chained things
        credit_entry.set_threshold(triggered.get_next_value());
        // Queue up remote credit DWQ
        libfab.queue_work(credit_entry.get_dwqe());

        // Queue up local completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        return TriggerStatus::NOT_NEEDED;
    }

    WaitStatus wait_gpu(hipStream_t* the_stream) override;

private:
    void* credit_buffers;

    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    // Counter trigger for credit counter
    CXICounter triggered;
    // Remote locations to send credit too
    std::array<ProtocolMatch::CreditBundle, Communication::MAX_CREDIT_SLACK>
        remote_credit_buffers;

    // Credit DWQ Preparations
    struct fid_cntr* completion_a;
    AtomicEntry      credit_entry;

    // Allocated Libfabric Objects
    struct fid_mr* my_mr;

    // User buffer details
    struct fi_rma_iov credit_buffer_rma_iov;

    // Local completion
    struct fid_cntr* completion_b;
    struct fid_cntr* completion_c;
    AtomicEntry      local_completion;
};

class CXIQueue : public HIPQueue
{
public:
    using CXIObjects = std::unique_ptr<CXIRequest>;

    CXIQueue(hipStream_t* stream_addr) : HIPQueue(stream_addr), comm_base(MPI_COMM_WORLD)
    {
        Print::out("CXI Queue init-ed");
        force_mpi(MPI_Comm_rank(comm_base, &my_rank));
        Print::out("Starting MPI Comm Dupes");
        force_mpi(MPI_Comm_dup(comm_base, &match_phase_a));
        force_mpi(MPI_Comm_dup(comm_base, &match_phase_b));
        Print::out("Starting Allreduce to get CXI address data");
        libfab.initialize(match_phase_a);
        Print::out("Creating on-node communicator");
        force_mpi(MPI_Comm_split_type(comm_base, MPI_COMM_TYPE_SHARED, my_rank,
                                      MPI_INFO_NULL, &on_node_peers));

        // Global counter for when requests can share a triggering counter
        the_gpu_counter = std::make_unique<CXICounter>(libfab);

        // Register MR
        my_buffer.register_mr(libfab);
    }

    ~CXIQueue()
    {
        MPI_Barrier(comm_base);
        MPI_Comm_free(&match_phase_a);
        MPI_Comm_free(&match_phase_b);
        MPI_Comm_free(&on_node_peers);
        the_gpu_counter.reset();
        request_map.clear();
        my_buffer.free_mr();
    }

    void enqueue_operation(std::shared_ptr<Request> request) override
    {
        enqueue_startall({request});
    }

    void enqueue_startall(std::vector<std::shared_ptr<Request>> requests) override;

    void enqueue_waitall() override;

    void host_wait() override
    {
        Queue::host_wait();
        Print::out("Waiting on device!");
        force_gpu(hipStreamSynchronize(*my_stream));
    }

    void initiate_match(std::vector<std::shared_ptr<Request>> requests) override
    {
        for (auto& req : requests)
        {
            if (Communication::Operation::BARRIER <= req->operation)
            {
                HIPQueue::initiate_match({req});
            }
            else
            {
                exchange_protocol(*req);
            }
        }
    }

    void finalize_match(std::vector<std::shared_ptr<Request>> requests) override
    {
        for (auto& req : requests)
        {
            if (Communication::Operation::BARRIER > req->operation)
            {
                prepare_cxi_mr_key(*req);
            }
        }
        Queue::finalize_match(requests);
    }

private:
    void exchange_protocol(Request&);
    void prepare_cxi_mr_key(Request&);
    void flush_memory();

    void inline start_request(CXIObjects& cxi_stuff) {}

    // Persistent Libfabric objects
    LibfabricInstance libfab;

    // Peer information
    MPI_Comm comm_base;
    int      my_rank;
    MPI_Comm match_phase_a;
    MPI_Comm match_phase_b;
    MPI_Comm on_node_peers;

    // Matching stuff
    std::map<size_t, std::unique_ptr<int>> active_match_requests;

    // Map of Request ID to CXIObject (counters, mr)
    std::map<size_t, CXIObjects> request_map;
    std::vector<size_t>          active_requests;

    // Completion buffers
    CompletionBufferFactory my_buffer;

    // GPU Triggerable Counter
    std::unique_ptr<CXICounter> the_gpu_counter;
};

#endif
