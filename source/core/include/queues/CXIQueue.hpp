
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

static inline void print_dfwq_entry(struct fi_deferred_work* dfwq_entry)
{
    Print::out("--- Start ---");
    Print::out("Threshold:", dfwq_entry->threshold);
    Print::out("Local IOVEC:", dfwq_entry->op.rma->msg.msg_iov->iov_base,
               dfwq_entry->op.rma->msg.msg_iov->iov_len);
    Print::out("Remote IOVEC:", dfwq_entry->op.rma->msg.rma_iov->addr,
               dfwq_entry->op.rma->msg.rma_iov->len,
               dfwq_entry->op.rma->msg.rma_iov->key);
    Print::out("--- End ---");
}

class CompletionBuffer
{
public:
    CompletionBuffer(void* _address, size_t _len, uint64_t _mr_key, uint64_t _offset)
        : address(_address), remote_description({_offset, _len, _mr_key})
    {
    }

    operator fi_rma_iov() const
    {
        return remote_description;
    }

    struct fi_rma_iov* get_rma_iov_addr()
    {
        return &remote_description;
    }

    virtual void print()
    {
        Print::out("Completion Buffer:", address, remote_description.addr,
                   remote_description.len, remote_description.key);
    }

    void*             address;
    struct fi_rma_iov remote_description;
};

class ProtocolBuffer : public CompletionBuffer
{
public:
    ProtocolBuffer(Operation _op, void* _address, size_t _len, uint64_t _mr_key,
                   uint64_t _offset)
        : CompletionBuffer(_address, _len, _mr_key, _offset), operation(_op)
    {
    }

    void print() override
    {
        Print::out("ProtocolBuffer:", operation, address, remote_description.addr,
                   remote_description.len, remote_description.key);
    }

    Operation operation;
};

class LibfabricInstance
{
public:
    LibfabricInstance() = default;
    ~LibfabricInstance();

    void initialize(MPI_Comm comm_base)
    {
        comm_size = -1;
        check_mpi(MPI_Comm_size(comm_base, &comm_size));
        initialize_libfabric();
        initialize_peer_addresses(comm_base);
    }

    struct fid_cntr* alloc_counter()
    {
        struct fid_cntr*    new_ctr;
        struct fi_cntr_attr cntr_attr = {
            .events   = FI_CNTR_EVENTS_COMP,
            .wait_obj = FI_WAIT_UNSPEC,
        };
        force_libfabric(fi_cntr_open(domain, &cntr_attr, &new_ctr, NULL));
        return new_ctr;
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
        print_dfwq_entry(work_entry);
        force_libfabric(fi_control(&domain->fid, FI_QUEUE_WORK, work_entry));
    }

    struct fi_info*    fi;       /*!< Provider's data and features */
    struct fid_fabric* fabric;   /*!< Represents the network */
    struct fid_domain* domain;   /*!< A subsection of the network */
    struct fid_av*     av;       /*!< Address vector for connections */
    struct fid_ep*     ep;       /*!< An endpoint */
    struct fid_cq*     txcq;     /*!< The transmit completion queue */
    struct fid_cq*     rxcq;     /*!< The receive completion queue */
    struct fid_cntr*   recv_ctr; /*!< The counters for receiving */

private:
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
};

class CXICounter
{
public:
    CXICounter(LibfabricInstance& libfab) : counter(libfab.alloc_counter())
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
        return CompletionBuffer(x, DEFAULT_ITEM_SIZE, fi_mr_key(my_mr), offset_value);
    }

    ProtocolBuffer alloc_protocol_buffer(Operation op)
    {
        if (current_index >= DEFAULT_ITEMS)
            throw std::runtime_error("Out of space for completion buffer");
        if (nullptr == my_mr)
            throw std::runtime_error("Buffer is not registered with libfabric");
        void*    x            = ((char*)buffer) + (sizeof(size_t) * current_index);
        uint64_t offset_value = current_index * DEFAULT_ITEM_SIZE;
        current_index++;
        return ProtocolBuffer(op, x, DEFAULT_ITEM_SIZE, fi_mr_key(my_mr), offset_value);
    }

    struct fid_mr* my_mr;
    void*          buffer;
    size_t         current_index;

    using COMPLETION_TYPE                     = size_t;
    static constexpr size_t DEFAULT_ITEMS     = 1000;
    static constexpr size_t DEFAULT_ITEM_SIZE = sizeof(COMPLETION_TYPE);
    static constexpr size_t DEFAULT_SIZE      = DEFAULT_ITEMS * DEFAULT_ITEM_SIZE;

    static constexpr int                MAX_COMP_VALUES = 100000;
    static std::vector<COMPLETION_TYPE> completion_addrs;
};

class Threshold
{
public:
    Threshold() : _value(0), _counter_value(0) {}

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
    size_t _value         = 0;
    size_t _counter_value = 0;
};

class DeferredWorkQueueEntry
{
public:
    DeferredWorkQueueEntry() {}

    struct fi_deferred_work* get_dwqe()
    {
        return &work_entry;
    }

    void set_completion_counter(fid_cntr* completion_counter)
    {
        work_entry.completion_cntr = completion_counter;
    }

    virtual void set_trigger_counter(CXICounter& trigger_cntr)
    {
        work_entry.triggering_cntr = trigger_cntr.counter;
    }

    virtual void set_threshold(Threshold& threshold)
    {
        work_entry.threshold = threshold.value();
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
        rma_work.msg.msg_iov   = &msg_iov;
        rma_work.msg.iov_count = 1;
        // To who are we going to
        rma_work.msg.addr = partner;
        // Setting up remote iov info (NO ACTUAL BUFFER DATA)
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

protected:
    struct fi_op_rma  rma_work;
    struct iovec      msg_iov;
    struct fi_rma_iov msg_rma_iov;
};

class DeferredWorkQueue
{
public:
    DeferredWorkQueue() = default;

    void register_progress_counter(struct fid_cntr* p_cntr)
    {
        Print::out("Currently in use:", space_used);
        progress_cntr = p_cntr;
    }

    void register_completion_counter(struct fid_cntr* c_cntr)
    {
        if (!known_completion_map.contains(c_cntr))
        {
            Print::out("Adding completion counter:", c_cntr);
            known_completion_map.insert({c_cntr, 0});
        }
    }

    void consume()
    {
        Print::out("*** Consumed");
        space_used++;
    }

    void clear_completion_counter(struct fid_cntr* completion_cntr,
                                  uint64_t         max_threshold)
    {
        Print::out("Clearing counter to:", max_threshold);
        uint64_t last_value = known_completion_map.at(completion_cntr);
        Print::out("[Counter, space] used before clearing:", last_value, space_used);

        uint64_t new_value = fi_cntr_read(completion_cntr);
        while (new_value != max_threshold)
        {
            progress();
            new_value = fi_cntr_read(completion_cntr);
        }

        // Cleanup
        space_used -= (new_value - last_value);
        Print::out("[Counter, space] used after clearing:", new_value, space_used);
        known_completion_map.erase(completion_cntr);
    }

    void make_space()
    {
        Print::out("*** Asked to make space at", space_used);
        if ((space_used + 1) >= total_space)
        {
            while ((space_used + 1) >= total_space)
            {
                progress();
                update_space_free();
            }
        }
    }

    uint64_t progress()
    {
        return fi_cntr_read(progress_cntr);
    }

    void print_status()
    {
        Print::always("Space in use:", space_used);
    }

private:
    inline void update_space_free()
    {
        for (auto& [counter, value] : known_completion_map)
        {
            uint64_t last_value = value;
            uint64_t curr_value = fi_cntr_read(counter);
            if (curr_value != last_value)
            {
                Print::out("*** Counter", counter, "was updated from", last_value, "to",
                           curr_value);
                space_used -= (curr_value - last_value);
                Print::out("*** Space in use:", space_used);
                value = curr_value;
            }
        }
    }

    // Control of DFWQ Space
    const uint64_t total_space = 84;
    uint64_t       space_used  = 0;

    // Progress counter
    struct fid_cntr* progress_cntr;

    // Last known completions
    std::map<struct fid_cntr*, uint64_t> known_completion_map;
};

class CXIRequest
{
public:
    CXIRequest(Request& req, CompletionBufferFactory& buffers)
        : base_req(req),
          completion_buffer(buffers.alloc_buffer()),
          protocol_buffer(buffers.alloc_protocol_buffer(req.operation)),
          num_times_started(0)
    {
    }

    virtual ~CXIRequest() = default;
    virtual void start(hipStream_t* the_stream, Threshold& threshold,
                       CXICounter& trigger_cntr)
    {
        num_times_started++;
        start_host(the_stream, threshold, trigger_cntr);
        start_gpu(the_stream, threshold, trigger_cntr);
        completion_buffer.print();
        protocol_buffer.print();
    }

    virtual void wait_gpu(hipStream_t* the_stream);

    virtual GPUMemoryType get_gpu_memory_type()
    {
        return base_req.get_memory_type();
    }

protected:
    virtual void start_host(hipStream_t* the_stream, Threshold& threshold,
                            CXICounter& trigger_cntr) = 0;
    virtual void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                           CXICounter& trigger_cntr)  = 0;

    Request&         base_req;
    CompletionBuffer completion_buffer;
    ProtocolBuffer   protocol_buffer;

    size_t num_times_started;
};

class FakeBarrier : public CXIRequest
{
public:
    // Uses "GPUMemoryType::FINE" because this doesn't need a flush
    FakeBarrier(Request& req, CompletionBufferFactory& buffers, DeferredWorkQueue& dwq)
        : CXIRequest(req, buffers), finished(true), progress_engine(dwq)
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

protected:
    void start_host(hipStream_t* the_stream, Threshold& threshold,
                    CXICounter& trigger_cntr) override
    {
        /* If previously launched, make do progress in case it's stuck */
        while (!finished)
        {
            /* Do progress (fi_cntr_read) */
            progress_engine.progress();
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
    DeferredWorkQueue& progress_engine;
};

template <bool FENCE = false>
class ChainedRMA
{
public:
    ChainedRMA(CompletionBuffer local_completion, struct fid_ep* ep, fi_addr_t partner,
               fi_addr_t self, struct fid_cntr* trigger, struct fid_cntr* remote_cntr,
               struct fid_cntr* local_cntr, void** comp_desc = nullptr)
    {
        Print::out("Address of completion values:",
                   CompletionBufferFactory::completion_addrs.data());
        chain_work_remote.threshold       = 0;
        chain_work_remote.triggering_cntr = trigger;
        chain_work_remote.completion_cntr = remote_cntr;
        chain_work_remote.op_type         = FI_OP_WRITE;

        chain_work_local.threshold       = 0;
        chain_work_local.triggering_cntr = remote_cntr;
        chain_work_local.completion_cntr = local_cntr;
        chain_work_local.op_type         = FI_OP_WRITE;

        chain_iovec = {};

        // RMA Send using offsets in remote buffer (not virtual addresses)
        /* Filled in by the type cast of Buffer class */
        local_rma_iov = local_completion;
        /* The first and last values should be filled in by the match! */
        remote_rma_iov = {0, CompletionBufferFactory::DEFAULT_ITEM_SIZE, 0};

        local_base_rma.ep  = ep;
        local_base_rma.msg = {
            .msg_iov       = &chain_iovec,
            .desc          = comp_desc,
            .iov_count     = 1,
            .addr          = self,
            .rma_iov       = &local_rma_iov,
            .rma_iov_count = 1,
            .context       = nullptr,
            .data          = 0,
        };

        remote_base_rma.ep  = ep;
        remote_base_rma.msg = {
            .msg_iov       = &chain_iovec,
            .desc          = comp_desc,
            .iov_count     = 1,
            .addr          = partner,
            .rma_iov       = &remote_rma_iov,
            .rma_iov_count = 1,
            .context       = nullptr,
            .data          = 0,
        };

        local_base_rma.flags =
            FI_DELIVERY_COMPLETE | ((FENCE) ? FI_FENCE : FI_CXI_WEAK_FENCE);
        remote_base_rma.flags =
            FI_DELIVERY_COMPLETE | ((FENCE) ? FI_FENCE : FI_CXI_WEAK_FENCE);
        chain_work_local.op.rma  = &local_base_rma;
        chain_work_remote.op.rma = &remote_base_rma;
    }
    void queue_work(LibfabricInstance& libfab)
    {
        /* Increase thresholds before starting! */
        chain_work_remote.threshold++;
        chain_work_local.threshold++;
        chain_iovec = {&CompletionBufferFactory::completion_addrs.at(index++),
                       CompletionBufferFactory::DEFAULT_ITEM_SIZE};

        libfab.queue_work(&chain_work_remote);
        libfab.queue_work(&chain_work_local);
    }

    struct fi_rma_iov* get_rma_iov_addr()
    {
        return &remote_rma_iov;
    }

    uint64_t get_threshold()
    {
        return chain_work_remote.threshold;
    }

private:
    struct fi_deferred_work chain_work_local;
    struct fi_op_rma        local_base_rma;
    struct fi_rma_iov       local_rma_iov;

    struct fi_deferred_work chain_work_remote;
    struct fi_op_rma        remote_base_rma;
    struct fi_rma_iov       remote_rma_iov;

    struct iovec chain_iovec;

    // Which completion value are we on?
    int index = 0;
};

class CXISend : public CXIRequest
{
public:
    CXISend(Request& user_request, CompletionBufferFactory& buffers,
            LibfabricInstance& _libfab, DeferredWorkQueue& dwq, fi_addr_t self)
        : CXIRequest(user_request, buffers),
          work_entry(_libfab.ep,
                     {user_request.buffer,
                      static_cast<size_t>(get_size_of_buffer(user_request))},
                     _libfab.get_peer(user_request.peer)),
          libfab(_libfab),
          my_queue(dwq),
          completion_a(_libfab.alloc_counter()),
          completion_b(_libfab.alloc_counter()),
          completion_c(_libfab.alloc_counter()),
          my_chained_completions(completion_buffer, _libfab.ep,
                                 _libfab.get_peer(user_request.peer), self, completion_a,
                                 completion_b, completion_c)
    {
        work_entry.set_completion_counter(completion_a);
        my_queue.register_completion_counter(completion_c);

        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::sender(
            work_entry.get_rma_iov_addr(), my_chained_completions.get_rma_iov_addr(),
            protocol_buffer.get_rma_iov_addr(), user_request);
    }

    ~CXISend()
    {
        my_queue.clear_completion_counter(completion_c,
                                          my_chained_completions.get_threshold());
        // Free counter
        force_libfabric(fi_close(&completion_a->fid));
        force_libfabric(fi_close(&completion_b->fid));
        force_libfabric(fi_close(&completion_c->fid));
    }

    void start_host(hipStream_t* the_stream, Threshold& threshold,
                    CXICounter& trigger_cntr) override
    {
        // Update threshold of chained things
        work_entry.set_threshold(threshold);
        // Adjust the triggering counter to use
        work_entry.set_trigger_counter(trigger_cntr);

        // Make sure we can add at 1 iteration (3 DFWQ entries technically)
        my_queue.make_space();

        libfab.queue_work(work_entry.get_dwqe());

        // Queue up chained actions
        my_chained_completions.queue_work(libfab);
        my_queue.consume();
    }

    void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                   CXICounter& trigger_cntr) override;

private:
    // Structs for the DFWQ Entry:
    RMAEntry work_entry;

    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    DeferredWorkQueue& my_queue;

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
          cts_entry(_libfab.ep, _libfab.get_peer(user_request.peer)),
          completion_a(_libfab.alloc_counter())
    {
        my_mr = _libfab.create_mr(user_request.buffer, get_size_of_buffer(user_request),
                                  FI_REMOTE_WRITE, FI_MR_ALLOCATED);

        user_buffer_rma_iov = {0, get_size_of_buffer(user_request), fi_mr_key(my_mr)};

        cts_entry.set_completion_counter(completion_a);

        /* Start requests to exchange from peer */
        Communication::ProtocolMatch::receiver(
            &user_buffer_rma_iov, completion_buffer.get_rma_iov_addr(),
            &protocol_buffer.operation, cts_entry.get_rma_iov_addr(), user_request);
    }

    ~CXIRecvOneSided()
    {
        // Free counter
        force_libfabric(fi_close(&completion_a->fid));
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
    }

    void start_host(hipStream_t* the_stream, Threshold& threshold,
                    CXICounter& trigger_cntr) override
    {
        if (Operation::RSEND != protocol_buffer.operation)
        {
            Print::out("Queue CTS to Libfabric!");
            // Update threshold of chained things
            cts_entry.set_threshold(threshold);
            // Adjust the triggering counter to use
            cts_entry.set_trigger_counter(trigger_cntr);
            cts_entry.set_iovec({&CompletionBufferFactory::completion_addrs.at(index++),
                                 CompletionBufferFactory::DEFAULT_ITEM_SIZE});
            libfab.queue_work(cts_entry.get_dwqe());
        }
    }

    void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                   CXICounter& trigger_cntr) override;

private:
    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    // CTS Preparations
    struct fid_cntr* completion_a;
    RMAEntry         cts_entry;
    // Which CTS value are we on?
    int index = 0;

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
        libfab.initialize(comm_base);
        the_gpu_counter = std::make_unique<CXICounter>(libfab);

        // Register MR
        my_buffer.register_mr(libfab);

        // Register progress counter
        my_queue.register_progress_counter(libfab.recv_ctr);
    }

    ~CXIQueue()
    {
        MPI_Barrier(comm_base);
        the_gpu_counter.reset();
        request_map.clear();
        my_buffer.free_mr();
    }

    void enqueue_operation(std::shared_ptr<Request> qe) override
    {
        queue_thresholds.increment_threshold();
        if (GPUMemoryType::COARSE == qe->get_memory_type())
        {
            flush_memory(the_stream);
        }
        enqueue_request(*qe);
    }

    void enqueue_startall(std::vector<std::shared_ptr<Request>> requests) override
    {
        bool shouldFlush = true;
        queue_thresholds.increment_threshold();
        for (auto& req : requests)
        {
            if (shouldFlush && GPUMemoryType::COARSE == req->get_memory_type())
            {
                flush_memory(the_stream);
                shouldFlush = false;
            }
            enqueue_request(*req);
        }
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
                qe->getID(), std::make_unique<FakeBarrier>(*qe, my_buffer, my_queue)));
        }
        else
        {
            prepare_cxi_mr_key(*qe);
        }
    }

private:
    void prepare_cxi_mr_key(Request&);
    void libfabric_teardown();
    void flush_memory(hipStream_t*);

    void inline enqueue_request(Request& req)
    {
        Print::out("Staring request:", req.getID());
        CXIObjects& cxi_stuff = request_map.at(req.getID());
        cxi_stuff->start(the_stream, queue_thresholds, *the_gpu_counter);

        // Keep track of active requests
        active_requests.push_back(req.getID());
        Print::out("... done");
    }

    // Persistent Libfabric objects
    LibfabricInstance libfab;

    // Peer information
    MPI_Comm comm_base;
    int      my_rank;

    // Map of Request ID to CXIObject (counters, mr)
    std::map<size_t, CXIObjects> request_map;
    std::vector<size_t>          active_requests;

    // Completion buffers
    static CompletionBufferFactory my_buffer;

    // Deferred Work Queue Management
    static DeferredWorkQueue my_queue;

    // Hip Stream
    hipStream_t* the_stream;
    // GPU Triggerable Counter
    std::unique_ptr<CXICounter> the_gpu_counter;
    Threshold                   queue_thresholds;
};

#endif
