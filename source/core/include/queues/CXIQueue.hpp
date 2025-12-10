
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
                   work_entry->completion_cntr);

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

    void enqueue_trigger(hipStream_t*);

    size_t get_next_value()
    {
        return use_count + 1;
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
    NOT_NEEDED  = 0,
    GLOBAL_BUMP = 1
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
    virtual TriggerStatus start(hipStream_t* the_stream, CXICounter& trigger_cntr)
    {
        num_times_started++;
        return start_derived(the_stream, trigger_cntr);
    }

    virtual void wait_gpu(hipStream_t* the_stream);

    virtual GPUMemoryType get_gpu_memory_type()
    {
        return base_req.get_memory_type();
    }

    virtual void match(MPI_Comm phase_a, MPI_Comm phase_b) = 0;

protected:
    virtual TriggerStatus start_derived(hipStream_t* the_stream,
                                        CXICounter&  trigger_cntr) = 0;

    Request&         base_req;
    CompletionBuffer completion_buffer;
    CompletionBuffer protocol_buffer;

    size_t num_times_started;
};

class FakeBarrier : public CXIRequest
{
public:
    FakeBarrier(Request& req, LibfabricInstance& _libfab)
        : CXIRequest(req), finished(true), progress_engine(_libfab)
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

protected:
    TriggerStatus start_derived(hipStream_t* the_stream,
                                CXICounter&  trigger_cntr) override
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
        thr      = std::thread(&FakeBarrier::thread_function, this, num_times_started);

        force_gpu(
            hipStreamWriteValue64(*the_stream, gpu_start_location, num_times_started, 0));
        return TriggerStatus::NOT_NEEDED;
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

class CXIRSend : public CXIRequest
{
public:
    CXIRSend(Request& user_request, CompletionBufferFactory& buffers,
             LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRequest(user_request, buffers),
          work_entry(_libfab.ep,
                     {user_request.buffer,
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
        Communication::ProtocolMatch::sender(work_entry.get_rma_iov_addr(),
                                             protocol_buffer.get_rma_ioc_addr(), base_req,
                                             comm_a, comm_b);
    }

    TriggerStatus start_derived(hipStream_t* the_stream,
                                CXICounter&  trigger_cntr) override
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

private:
    // Structs for the DFWQ Entry:
    RMAEntry    work_entry;
    AtomicEntry local_completion;

    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    struct fid_cntr* completion_a;
    struct fid_cntr* completion_b;
    struct fid_cntr* completion_c;
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
          local_completion(_libfab.ep, self, completion_buffer.get_rma_ioc_addr()),
          libfab(_libfab),
          completion_a(_libfab.alloc_counter(true)),
          completion_b(_libfab.alloc_counter(true)),
          triggered(_libfab)
    {
        work_entry.set_trigger_counter(triggered);
        work_entry.set_completion_counter(completion_a);

        local_completion.set_trigger_counter(completion_a);
        local_completion.set_completion_counter(completion_b);

        /* Setup CTS buffer */
        force_gpu(hipHostMalloc(&cts_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE,
                                hipHostMallocDefault));
        cts_mr = _libfab.create_mr_with_counter(
            cts_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE, FI_REMOTE_WRITE,
            FI_MR_ALLOCATED, triggered.counter, FI_REMOTE_WRITE);

        /* Set Protocol Buffer */
        protocol_buffer =
            CompletionBuffer(cts_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE, 1,
                             fi_mr_key(cts_mr), 0);
    }

    ~CXISend()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_b);
        check_gpu(hipHostFree(cts_buffer));
        check_libfabric(fi_close(&(cts_mr)->fid));
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::sender(work_entry.get_rma_iov_addr(),
                                             protocol_buffer.get_rma_ioc_addr(), base_req,
                                             comm_a, comm_b);
    }

    TriggerStatus start_derived(hipStream_t* the_stream,
                                CXICounter&  trigger_cntr) override
    {
        // Update threshold of chained things
        work_entry.set_threshold(triggered.get_next_value() * 2);

        // Queue up send of user data
        // work_entry.print();
        libfab.queue_work(work_entry.get_dwqe());

        // Queue up completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        triggered.enqueue_trigger(the_stream);
        return TriggerStatus::GLOBAL_BUMP;
    }

private:
    // Structs for the DFWQ Entry:
    RMAEntry    work_entry;
    AtomicEntry local_completion;

    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    struct fid_cntr* completion_a;
    struct fid_cntr* completion_b;

    CXICounter triggered;
    void*      cts_buffer;
    fid_mr*    cts_mr;
};

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
            user_request.buffer, get_size_of_buffer(user_request), FI_REMOTE_WRITE,
            FI_MR_ALLOCATED, completion_b, FI_REMOTE_WRITE);

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
        Communication::ProtocolMatch::receiver(&user_buffer_rma_iov, &peer_op,
                                               cts_entry.get_rma_ioc_addr(), base_req,
                                               comm_a, comm_b);
    }

    TriggerStatus start_derived(hipStream_t* the_stream,
                                CXICounter&  trigger_cntr) override
    {
        TriggerStatus rc = TriggerStatus::NOT_NEEDED;
        if (Operation::RSEND != peer_op)
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
    Operation        peer_op;

    // Allocated Libfabric Objects
    struct fid_mr* my_mr;

    // User buffer details
    struct fi_rma_iov user_buffer_rma_iov;

    // Local completion
    struct fid_cntr* completion_b;
    struct fid_cntr* completion_c;
    AtomicEntry      local_completion;
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
        enqueue_startall({request});
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

        bool should_bump = false;
        for (auto& req : requests)
        {
            CXIObjects& cxi_req = request_map.at(req->getID());
            /* Start Request */
            Print::out("Staring request:", req->getID());
            TriggerStatus actions = cxi_req->start(the_stream, *the_gpu_counter);
            Print::out("Request request:", actions);

            /* Add request to be waited on. */
            active_requests.push_back(req->getID());

            /* Enqueue global counter bump, if needed */
            if (!should_bump && TriggerStatus::GLOBAL_BUMP == actions)
            {
                should_bump = true;
            }
        }
        if (should_bump)
        {
            the_gpu_counter->enqueue_trigger(the_stream);
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
            request_map.insert(
                std::make_pair(qe->getID(), std::make_unique<FakeBarrier>(*qe, libfab)));
        }
        else
        {
            prepare_cxi_mr_key(*qe);
        }
    }

private:
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

    // Map of Request ID to CXIObject (counters, mr)
    std::map<size_t, CXIObjects> request_map;
    std::vector<size_t>          active_requests;

    // Completion buffers
    static CompletionBufferFactory my_buffer;

    // Hip Stream
    hipStream_t* the_stream;
    // GPU Triggerable Counter
    std::unique_ptr<CXICounter> the_gpu_counter;
};

#endif
