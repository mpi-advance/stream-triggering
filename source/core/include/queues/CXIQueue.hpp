
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
#include "abstract/queue.hpp"
#include "misc/print.hpp"
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
        struct fid_mr* new_mr = nullptr;
        force_libfabric(
            fi_mr_reg(domain, buffer, len, access, 0, getMRID(), flags, &new_mr, NULL));
        force_libfabric(fi_mr_bind(new_mr, &(ep)->fid, 0));

        // Enable MR
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
        Print::out("<H> Threshold:", work_entry->threshold, work_entry->triggering_cntr);

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

    ~CXICounter()
    {
        // Free counter
        force_libfabric(fi_close(&counter->fid));
        force_gpu(hipHostUnregister(mmio_addr));
    }

    void print()
    {
        size_t value = fi_cntr_read(counter);
        Print::out("Value: ", value);
    }

    // Libfabric Structs
    struct fid_cntr*        counter;
    struct fi_cxi_cntr_ops* counter_ops;
    // MMIO Pointers
    void*  mmio_addr;
    size_t mmio_addr_len;
    void*  gpu_mmio_addr;
};

class CompletionBufferFactory
{
public:
    CompletionBufferFactory() : my_mr(nullptr)
    {
        force_gpu(hipHostMalloc(&buffer, DEFAULT_SIZE, hipHostMallocDefault));
        Print::out("Default Completion Buffer location:", buffer);
        memset(buffer, 0, DEFAULT_SIZE);
    }

    ~CompletionBufferFactory()
    {
        if (my_mr)
        {
            force_libfabric(fi_close(&(my_mr)->fid));
        }
        check_gpu(hipHostFree(buffer));
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

    struct fid_mr* my_mr;
    void*          buffer;
    size_t         current_index;

    using COMPLETION_TYPE                     = size_t;
    static constexpr size_t DEFAULT_ITEMS     = 1000;
    static constexpr size_t DEFAULT_ITEM_SIZE = sizeof(COMPLETION_TYPE);
    static constexpr size_t DEFAULT_SIZE      = DEFAULT_ITEMS * DEFAULT_ITEM_SIZE;
};

class Threshold
{
public:
    Threshold() : _value(1), _counter_value(0) {}

    void increment_threshold()
    {
        _value++;
    }

    size_t equalize_counter()
    {
        size_t diff    = _value - _counter_value;
        _counter_value = _value;
        return diff;
    }

    size_t value()
    {
        return _value;
    }
    size_t counter_value()
    {
        return _counter_value;
    }

private:
    size_t _value         = 1;
    size_t _counter_value = 0;
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

    virtual void set_threshold(Threshold& threshold)
    {
        work_entry.threshold = threshold.value();
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

    void set_rma_iov(struct fi_rma_iov new_rma_iov)
    {
        msg_rma_iov = new_rma_iov;
    }

    void set_flags(uint64_t flags)
    {
        rma_work.flags = flags;
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

    void set_rma_iov(struct fi_rma_iov new_rma_iov)
    {
        msg_rma_ioc = {new_rma_iov.addr, 1, new_rma_iov.key};
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

enum TriggerStatus
{
    NOT_NEEDED = 0,
    READY      = 1,
    NEEDS_CNTR = 2,
    QUEUED     = 3
};

class CXIRequest
{
public:
    CXIRequest(Request& req, CompletionBufferFactory& buffers)
        : base_req(req),
          completion_buffer(buffers.alloc_buffer()),
          protocol_buffer(buffers.alloc_buffer()),
          num_times_started(0)
    {
        Print::out("Request ID: ", req.getID());
        // completion_buffer.print();
        // protocol_buffer.print();
    }

    virtual ~CXIRequest() = default;
    virtual void start(hipStream_t* the_stream, Threshold& threshold,
                       CXICounter& trigger_cntr)
    {
        num_times_started++;
        start_host(the_stream, threshold, trigger_cntr);
        start_gpu(the_stream, threshold, trigger_cntr);
    }

    virtual void wait_gpu(hipStream_t* the_stream);

    virtual GPUMemoryType get_gpu_memory_type()
    {
        return base_req.get_memory_type();
    }

    virtual void match(MPI_Comm phase_a, MPI_Comm phase_b) = 0;

    virtual TriggerStatus getTriggerStatus() = 0;

protected:
    virtual void start_host(hipStream_t* the_stream, Threshold& threshold,
                            CXICounter& trigger_cntr) = 0;
    virtual void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                           CXICounter& trigger_cntr)  = 0;

    Request&         base_req;
    CompletionBuffer completion_buffer;
    CompletionBuffer protocol_buffer;

    size_t num_times_started;
};

class FakeBarrier : public CXIRequest
{
public:
    FakeBarrier(Request& req, CompletionBufferFactory& buffers,
                LibfabricInstance& _libfab)
        : CXIRequest(req, buffers), finished(true), progress_engine(_libfab)
    {
        // Setup GPU memory locations
        force_gpu(hipHostMalloc((void**)&host_start_location, sizeof(int64_t),
                                hipHostMallocDefault));
        *host_start_location = 0;
        force_gpu(hipHostGetDevicePointer(&gpu_start_location, host_start_location, 0));
        force_gpu(hipHostMalloc((void**)&host_wait_location, sizeof(int64_t),
                                hipHostMallocDefault));
        *host_wait_location = 0;
        force_gpu(hipHostGetDevicePointer(&gpu_wait_location, host_wait_location, 0));
    }

    ~FakeBarrier()
    {
        thr.join();
        check_gpu(hipHostFree(host_start_location));
        check_gpu(hipHostFree(host_wait_location));
    }

    void wait_gpu(hipStream_t* the_stream) override
    {
        force_gpu(
            hipStreamWaitValue64(*the_stream, gpu_wait_location, num_times_started, 0));
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override {}

    TriggerStatus getTriggerStatus() override
    {
        return TriggerStatus::READY;
    }

protected:
    void start_host(hipStream_t* the_stream, Threshold& threshold,
                    CXICounter& trigger_cntr) override
    {
        /* If previously launched, make do progress in case it's stuck */
        while (!finished)
        {
            /* Do progress (fi_cntr_read) */
            progress_engine.progress_dwq();
        }

        /* And normal thread joining check */
        if (thr.joinable())
        {
            thr.join();
        }
        /* Launch thread */
        finished = false;
        thr      = std::thread(&FakeBarrier::thread_function, this, threshold.value());
    }

    void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                   CXICounter& trigger_cntr) override
    {
        force_gpu(
            hipStreamWriteValue64(*the_stream, gpu_start_location, threshold.value(), 0));
    }

private:
    void thread_function(size_t thread_threshold)
    {
        /* Wait for signal from GPU */
        while (__atomic_load_n(host_start_location, __ATOMIC_ACQUIRE) < thread_threshold)
        {
            // Do nothing
        }

        /* Execute MPI Call */
        MPI_Barrier(base_req.comm);

        /* Mark completion location */
        (*host_wait_location) = num_times_started;
        /* End thread */
        finished = true;
    }

    // Memory locations
    size_t* host_start_location;
    size_t* host_wait_location;

    void* gpu_start_location;
    void* gpu_wait_location;

    // Thread
    std::thread thr;
    bool        finished = true;
    // Progress
    LibfabricInstance& progress_engine;
};

template <bool FENCE = false>
class ChainedRMA
{
public:
    ChainedRMA(struct fi_rma_ioc* local_completion, struct fid_ep* ep, fi_addr_t partner,
               fi_addr_t self, struct fid_cntr* trigger, struct fid_cntr* remote_cntr,
               struct fid_cntr* local_cntr, void** comp_desc = nullptr)
        : chain_work_remote(ep, partner), chain_work_local(ep, self, local_completion)
    {
        chain_work_remote.set_trigger_counter(trigger);
        chain_work_remote.set_completion_counter(remote_cntr);

        chain_work_local.set_trigger_counter(remote_cntr);
        chain_work_local.set_completion_counter(local_cntr);

        /* The first and last values should be filled in by the match! */
        chain_work_remote.set_rma_iov({0, 1, 0});

        chain_work_local.set_flags(FI_DELIVERY_COMPLETE |
                                   ((FENCE) ? FI_FENCE : FI_CXI_WEAK_FENCE));
        chain_work_remote.set_flags(FI_DELIVERY_COMPLETE |
                                    ((FENCE) ? FI_FENCE : FI_CXI_WEAK_FENCE));
    }
    void queue_work(LibfabricInstance& libfab)
    {
        /* Increase thresholds before starting! */
        chain_work_remote.bump_threshold();
        chain_work_local.bump_threshold();

        // chain_work_remote.print();
        libfab.queue_work(chain_work_remote.get_dwqe());
        // chain_work_local.print();
        libfab.queue_work(chain_work_local.get_dwqe());
    }

    struct fi_rma_ioc* get_remote_rma_ioc_addr()
    {
        return chain_work_remote.get_rma_ioc_addr();
    }

private:
    AtomicEntry chain_work_local;
    AtomicEntry chain_work_remote;

    // Which completion value are we on?
    int index = 0;
};

class CXISend : public CXIRequest
{
public:
    CXISend(Request& user_request, CompletionBufferFactory& buffers,
            LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRequest(user_request, buffers),
          work_entry(_libfab.ep,
                     {user_request.buffer,
                      static_cast<size_t>(get_size_of_buffer(user_request))},
                     _libfab.get_peer(user_request.resolve_comm_world())),
          libfab(_libfab),
          completion_a(_libfab.alloc_counter(true)),
          completion_b(_libfab.alloc_counter(true)),
          completion_c(_libfab.alloc_counter(true)),
          my_chained_completions(completion_buffer.get_rma_ioc_addr(), _libfab.ep,
                                 _libfab.get_peer(user_request.resolve_comm_world()),
                                 self, completion_a, completion_b, completion_c)
    {
        work_entry.set_completion_counter(completion_a);
        work_entry.set_flags(FI_DELIVERY_COMPLETE | FI_CXI_WEAK_FENCE);
    }

    ~CXISend()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_b);
        libfab.dealloc_counter(completion_c);
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::sender(
            work_entry.get_rma_iov_addr(),
            my_chained_completions.get_remote_rma_ioc_addr(),
            protocol_buffer.get_rma_ioc_addr(), base_req, comm_a, comm_b);
    }

    void start_host(hipStream_t* the_stream, Threshold& threshold,
                    CXICounter& trigger_cntr) override
    {
        // Update threshold of chained things
        work_entry.set_threshold(threshold);
        // Adjust the triggering counter to use
        work_entry.set_trigger_counter(trigger_cntr);

        // Queue up send of user data
        // work_entry.print();
        libfab.queue_work(work_entry.get_dwqe());

        // Queue up chained actions
        my_chained_completions.queue_work(libfab);
    }

    void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                   CXICounter& trigger_cntr) override;

    TriggerStatus getTriggerStatus() override
    {
        return (Operation::SEND == base_req.operation) ? TriggerStatus::NEEDS_CNTR
                                                       : TriggerStatus::READY;
    }

private:
    // Structs for the DFWQ Entry:
    RMAEntry work_entry;

    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    struct fid_cntr*  completion_a;
    struct fid_cntr*  completion_b;
    struct fid_cntr*  completion_c;
    ChainedRMA<false> my_chained_completions;
};

class CXIRecvOneSided : public CXIRequest
{
public:
    CXIRecvOneSided(Request& user_request, CompletionBufferFactory& buffers,
                    LibfabricInstance& _libfab)
        : CXIRequest(user_request, buffers),
          libfab(_libfab),
          cts_entry(_libfab.ep, _libfab.get_peer(user_request.resolve_comm_world())),
          completion_a(_libfab.alloc_counter(true))
    {
        my_mr = _libfab.create_mr(user_request.buffer, get_size_of_buffer(user_request),
                                  FI_REMOTE_WRITE, FI_MR_ALLOCATED);

        user_buffer_rma_iov = {0, get_size_of_buffer(user_request), fi_mr_key(my_mr)};

        cts_entry.set_completion_counter(completion_a);
        cts_entry.set_flags(FI_DELIVERY_COMPLETE | FI_CXI_WEAK_FENCE);
    }

    ~CXIRecvOneSided()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::receiver(
            &user_buffer_rma_iov, completion_buffer.get_rma_ioc_addr(), &peer_op,
            cts_entry.get_rma_ioc_addr(), base_req, comm_a, comm_b);
    }

    void start_host(hipStream_t* the_stream, Threshold& threshold,
                    CXICounter& trigger_cntr) override
    {
        if (Operation::RSEND != peer_op)
        {
            Print::out("Queue CTS to Libfabric!");
            // Update threshold of chained things
            cts_entry.set_threshold(threshold);
            // Adjust the triggering counter to use
            cts_entry.set_trigger_counter(trigger_cntr);
            libfab.queue_work(cts_entry.get_dwqe());
        }
    }

    void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                   CXICounter& trigger_cntr) override {};

    TriggerStatus getTriggerStatus() override
    {
        return (Operation::RSEND != peer_op) ? TriggerStatus::READY
                                             : TriggerStatus::NOT_NEEDED;
    }

private:
    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    // CTS Preparations
    struct fid_cntr* completion_a;
    AtomicEntry      cts_entry;
    Operation        peer_op;

    // Allocated Libfabric Objects
    struct fid_mr* my_mr;

    // User buffer details
    struct fi_rma_iov user_buffer_rma_iov;
};

class CXIQueue : public Queue
{
public:
    using CXIObjects = std::unique_ptr<CXIRequest>;

    CXIQueue(hipStream_t* stream_addr)
        : comm_base(MPI_COMM_WORLD), the_stream(stream_addr)
    {
        Print::out("CXI Queue init-ed");
        force_mpi(MPI_Comm_rank(comm_base, &my_rank));
        Print::out("Starting MPI Comm Dupes");
        force_mpi(MPI_Comm_dup(comm_base, &match_phase_a));
        force_mpi(MPI_Comm_dup(comm_base, &match_phase_b));
        Print::out("Starting Allreduce to get CXI address data");
        libfab.initialize(match_phase_a);
        the_gpu_counter = std::make_unique<CXICounter>(libfab);

        // Register MR
        my_buffer.register_mr(libfab);
    }

    ~CXIQueue()
    {
        MPI_Barrier(comm_base);
        MPI_Comm_free(&match_phase_a);
        MPI_Comm_free(&match_phase_b);
        the_gpu_counter.reset();
        request_map.clear();
        my_buffer.free_mr();
    }

    void enqueue_operation(std::shared_ptr<Request> request) override
    {
        std::vector<std::shared_ptr<Request>> temp(1);
        temp[0] = request;
        enqueue_startall(temp);
    }

    void enqueue_startall(std::vector<std::shared_ptr<Request>> requests) override
    {
#if defined(USE_TIOGA)
        /* If any kernel needs a flush, go ahead and do it first */
        for (auto& req : requests)
        {
            if (req->needs_gpu_flush())
            {
                flush_memory();
                break;
            }
        }
#endif

        bool needs_final_trigger = false;
        for (auto& req : requests)
        {
            CXIObjects&   cxi_req = request_map.at(req->getID());
            TriggerStatus actions = cxi_req->getTriggerStatus();
            Print::out("Staring request:", req->getID(), actions);
            switch (actions)
            {
                case TriggerStatus::NOT_NEEDED:
                    /* Request needs to know it was started*/
                    cxi_req->start(the_stream, queue_thresholds, *the_gpu_counter);
                    break;
                case TriggerStatus::READY:
                    /* Setup libfabric entries */
                    cxi_req->start(the_stream, queue_thresholds, *the_gpu_counter);
                    needs_final_trigger = true;
                    break;
                case TriggerStatus::NEEDS_CNTR:
                    /* If there are other un-triggered things queued up */
                    if (needs_final_trigger)
                    {
                        /* Queue trigger and increment threshold. */
                        enqueue_trigger();
                        queue_thresholds.increment_threshold();
                        needs_final_trigger = false;
                    }
                    /* Request does two things here:
                     * 1. Queues DWQ entry using the threshold and counter provided
                     * 2. Starts any extra GPU kernels, but not a triggering kernel */
                    cxi_req->start(the_stream, queue_thresholds, *the_gpu_counter);
                    /* Enqueue the kernel that will bump the counter. */
                    enqueue_trigger();
                    /* Set the threshold for next request. */
                    queue_thresholds.increment_threshold();
                    break;
                default:
                    throw std::runtime_error("Invalid Request State:" +
                                             std::to_string((int)actions));
            }
            /* Add request to be waited on. */
            active_requests.push_back(req->getID());
        }
        if (needs_final_trigger)
        {
            /* Queue trigger and increment threshold. */
            enqueue_trigger();
            queue_thresholds.increment_threshold();
        }
        Print::out("Done staring all requests.");
    }

    void enqueue_waitall() override;

    void host_wait() override
    {
        Print::out("Waiting on device!");
        force_gpu(hipStreamSynchronize(*the_stream));
    }

    void match(std::shared_ptr<Request> qe) override
    {
        if (Communication::Operation::BARRIER == qe->operation)
        {
            /* Add request to map */
            request_map.insert(std::make_pair(
                qe->getID(), std::make_unique<FakeBarrier>(*qe, my_buffer, libfab)));
        }
        else
        {
            prepare_cxi_mr_key(*qe);
        }
    }

private:
    void prepare_cxi_mr_key(Request&);
    void flush_memory();
    void enqueue_trigger();

    void inline start_request(CXIObjects& cxi_stuff) {}

    // Persistent Libfabric objects
    LibfabricInstance libfab;

    // Peer information
    MPI_Comm comm_base;
    int      my_rank;
    MPI_Comm match_phase_a;
    MPI_Comm match_phase_b;

    // Map of Request ID to CXIObject (counters, mr)
    std::map<size_t, CXIObjects> request_map;
    std::vector<size_t>          active_requests;

    // Completion buffers
    static CompletionBufferFactory my_buffer;

    // Hip Stream
    hipStream_t* the_stream;
    // GPU Triggerable Counter
    std::unique_ptr<CXICounter> the_gpu_counter;
    Threshold                   queue_thresholds;
};

#endif
