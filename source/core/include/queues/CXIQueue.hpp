
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

#include "abstract/queue.hpp"
#include "misc/print.hpp"
#include "safety/hip.hpp"
#include "safety/libfabric.hpp"
#include "safety/mpi.hpp"

class Buffer
{
public:
    Buffer(void* _address, size_t _len, uint64_t _mr_key, uint64_t _offset)
        : address(_address), len(_len), mr_key(_mr_key), offset(_offset)
    {
    }

    operator fi_rma_iov() const
    {
        return {offset, len, mr_key};
    }

    void print()
    {
        Print::out("Buffer:", address, len, mr_key, offset);
    }

    void*    address;
    size_t   len;
    uint64_t mr_key;
    uint64_t offset;
};

class DeferredWorkQueue
{
public:
    DeferredWorkQueue() = default;

    void regsiter_counter(struct fid_cntr* p_cntr)
    {
        progress_cntr = p_cntr;
    }

    void consume()
    {
        space_used++;
    }

    void make_space(struct fid_cntr* completion_cntr, uint64_t space_amount = 1)
    {
        if ((space_used + space_amount) >= total_space)
        {
            uint64_t last_value = 0;
            if (known_completion_map.contains(completion_cntr))
            {
                last_value = known_completion_map.at(completion_cntr);
            }
            else
            {
                known_completion_map.insert({completion_cntr, 0});
            }

            uint64_t curr_completed = fi_cntr_read(completion_cntr);
            while ((curr_completed - last_value) < space_amount)
            {
                progress();
                curr_completed = fi_cntr_read(completion_cntr);
            }

            space_used -= (curr_completed - last_value);
            known_completion_map.at(completion_cntr) = curr_completed;
        }
    }

    uint64_t progress()
    {
        return fi_cntr_read(progress_cntr);
    }

private:
    // Control of DFWQ Space
    const uint64_t total_space = 84;
    uint64_t       space_used  = 0;

    // Progress counter
    struct fid_cntr* progress_cntr;

    // Last known completions
    std::map<struct fid_cntr*, uint64_t> known_completion_map;
};

class CXICounter
{
public:
    static struct fid_cntr* alloc_counter(struct fid_domain* domain)
    {
        struct fid_cntr*    new_ctr;
        struct fi_cntr_attr cntr_attr = {
            .events   = FI_CNTR_EVENTS_COMP,
            .wait_obj = FI_WAIT_UNSPEC,
        };
        force_libfabric(fi_cntr_open(domain, &cntr_attr, &new_ctr, NULL));
        return new_ctr;
    }

    CXICounter(struct fid_domain* domain) : counter(alloc_counter(domain))
    {
        // Open (create) CXI Extension object
        check_libfabric(fi_open_ops(&(counter->fid), FI_CXI_COUNTER_OPS, 0,
                                    (void**)&counter_ops, NULL));
        // Get the MMIO Address of the counter
        check_libfabric(counter_ops->get_mmio_addr(&counter->fid, &mmio_addr,
                                                   &mmio_addr_len));
        // Register MMIO Address w/ HIP
        force_hip(
            hipHostRegister(mmio_addr, mmio_addr_len, hipHostRegisterDefault));
        // Get GPU version of MMIO address
        force_hip(hipHostGetDevicePointer(&gpu_mmio_addr, mmio_addr, 0));
    }

    ~CXICounter()
    {
        // Free counter
        force_libfabric(fi_close(&counter->fid));
        force_hip(hipHostUnregister(mmio_addr));
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

// MR Management
static size_t getMRID()
{
    static size_t ID = 1;
    return ID++;
}

class CompletionBuffer
{
public:
    CompletionBuffer() : my_mr(nullptr)
    {
        force_hip(hipHostMalloc(&buffer, DEFAULT_SIZE, hipHostMallocDefault));
    }

    ~CompletionBuffer()
    {
        if (my_mr)
        {
            force_libfabric(fi_close(&(my_mr)->fid));
        }
        check_hip(hipHostFree(buffer));
    }

    CompletionBuffer(const CompletionBuffer& other) = delete;
    CompletionBuffer(CompletionBuffer&& other)
    {
        buffer       = other.buffer;
        my_mr        = other.my_mr;
        other.buffer = nullptr;
        other.my_mr  = nullptr;
    }

    CompletionBuffer& operator=(const CompletionBuffer& rhs) = delete;
    CompletionBuffer& operator=(CompletionBuffer&& other)
    {
        buffer       = other.buffer;
        my_mr        = other.my_mr;
        other.buffer = nullptr;
        other.my_mr  = nullptr;
        return *this;
    }

    void register_mr(struct fid_domain* domain, struct fid_ep* main_ep)
    {
        force_libfabric(fi_mr_reg(domain, buffer, DEFAULT_SIZE,
                                  FI_REMOTE_WRITE | FI_WRITE, 0, getMRID(),
                                  FI_MR_ALLOCATED, &my_mr, NULL));
        force_libfabric(fi_mr_bind(my_mr, &(main_ep)->fid, 0));

        // Enable MR
        force_libfabric(fi_mr_enable(my_mr));
    }

    void free_mr()
    {
        check_libfabric(fi_close(&(my_mr)->fid));
        my_mr = nullptr;
    }

    Buffer alloc_buffer()
    {
        if (current_index >= DEFAULT_ITEMS)
            throw std::runtime_error("Out of space for completion buffer");
        if (nullptr == my_mr)
            throw std::runtime_error("Buffer is not registered with libfabric");
        void*    x = ((char*)buffer) + (sizeof(size_t) * current_index);
        uint64_t offset_value = current_index * DEFAULT_ITEM_SIZE;
        current_index++;
        return Buffer(x, DEFAULT_SIZE, fi_mr_key(my_mr), offset_value);
    }

    struct fid_mr* my_mr;
    void*          buffer;
    size_t         current_index;

    static constexpr size_t DEFAULT_ITEMS     = 1000;
    static constexpr size_t DEFAULT_ITEM_SIZE = sizeof(size_t);
    static constexpr size_t DEFAULT_SIZE = DEFAULT_ITEMS * DEFAULT_ITEM_SIZE;
};

class CXIRequest
{
public:
    CXIRequest(Buffer local_completion_buffer)
        : completion_buffer(local_completion_buffer)
    {
    }
    virtual ~CXIRequest() = default;
    virtual void start(hipStream_t* the_stream)
    {
        threshold++;
        start_host();
        start_gpu(the_stream);
    }
    virtual void wait_gpu(hipStream_t* the_stream) = 0;

protected:
    virtual void start_host()                       = 0;
    virtual void start_gpu(hipStream_t* the_stream) = 0;

    Buffer completion_buffer;
    size_t threshold = 0;
};

class FakeBarrier : public CXIRequest
{
public:
    FakeBarrier(Buffer buffer, MPI_Comm comm, DeferredWorkQueue& dwq)
        : CXIRequest(buffer),
          comm_to_use(comm),
          finished(true),
          progress_engine(dwq)
    {
        // Setup GPU memory locations
        force_hip(hipHostMalloc((void**)&host_start_location, sizeof(int64_t),
                                hipHostMallocDefault));
        *host_start_location = 0;
        force_hip(hipHostGetDevicePointer(&gpu_start_location,
                                          host_start_location, 0));
        force_hip(hipHostMalloc((void**)&host_wait_location, sizeof(int64_t),
                                hipHostMallocDefault));
        *host_wait_location = 0;
        force_hip(
            hipHostGetDevicePointer(&gpu_wait_location, host_wait_location, 0));
    }

    ~FakeBarrier()
    {
        thr.join();
        check_hip(hipHostFree(host_start_location));
        check_hip(hipHostFree(host_wait_location));
    }

    void wait_gpu(hipStream_t* the_stream) override
    {
        force_hip(
            hipStreamWaitValue64(*the_stream, gpu_wait_location, threshold, 0));
    }

protected:
    void start_host() override
    {
        /* If previously launched, make sure we do progress in case it's stuck
         */
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
        thr      = std::thread(&FakeBarrier::thread_function, this, threshold);
    }

    void start_gpu(hipStream_t* the_stream) override
    {
        force_hip(hipStreamWriteValue64(*the_stream, gpu_start_location,
                                        threshold, 0));
    }

private:
    void thread_function(int thread_threshold)
    {
        /* Wait for signal from GPU */
        while (__atomic_load_n(host_start_location, __ATOMIC_ACQUIRE) <
               thread_threshold)
        {
            // Do nothing
        }

        /* Execute MPI Call */
        MPI_Barrier(comm_to_use);

        /* Mark completion location */
        (*host_wait_location) = thread_threshold;
        /* End thread */
        finished = true;
    }

    // Memory locations
    size_t* host_start_location;
    size_t* host_wait_location;

    void* gpu_start_location;
    void* gpu_wait_location;
    // MPI Variables
    MPI_Comm comm_to_use;
    // Thread
    std::thread thr;
    bool        finished = true;
    // Progress
    DeferredWorkQueue& progress_engine;
};

class CXIWait : virtual public CXIRequest
{
public:
    void wait_gpu(hipStream_t* the_stream) override;
};

enum GPUMemoryType
{
    COARSE = 1,
    FINE   = 2,
};
template <GPUMemoryType G>
class CXITrigger : virtual public CXIRequest
{
public:
    CXITrigger(struct fid_domain* domain) : trigger_counter(domain) {}

    void start_gpu(hipStream_t* the_stream) override;

    struct fid_cntr* get_libfabric_counter()
    {
        return trigger_counter.counter;
    }

    void print_counter()
    {
        trigger_counter.print();
    }

private:
    CXICounter trigger_counter;
};

template <bool FENCE = false>
class ChainedRMA
{
public:
    ChainedRMA(Buffer local_completion, Buffer remote_completion,
               struct fid_ep* ep, fi_addr_t partner, fi_addr_t self,
               struct fid_cntr* trigger, struct fid_cntr* remote_cntr,
               struct fid_cntr* local_cntr, void** comp_desc = nullptr)
        : completion_addrs(MAX_COMP_VALUES)
    {
        std::iota(completion_addrs.begin(), completion_addrs.end(), 1);
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
        local_rma_iov  = local_completion;   // Type cast of Buffer class
        remote_rma_iov = remote_completion;  // Type cast of Buffer class

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

    void set_active_threshold(int threshold)
    {
        chain_work_remote.threshold = threshold;
        chain_work_local.threshold  = threshold;
        chain_iovec = {&completion_addrs.at(index++), sizeof(int)};
    }

    void queue_work(struct fid_domain* domain)
    {
        check_libfabric(
            fi_control(&domain->fid, FI_QUEUE_WORK, &chain_work_remote));
        check_libfabric(
            fi_control(&domain->fid, FI_QUEUE_WORK, &chain_work_local));
    }

private:
    struct fi_deferred_work chain_work_local;
    struct fi_op_rma        local_base_rma;
    struct fi_rma_iov       local_rma_iov;

    struct fi_deferred_work chain_work_remote;
    struct fi_op_rma        remote_base_rma;
    struct fi_rma_iov       remote_rma_iov;

    struct iovec chain_iovec;

    static constexpr int MAX_COMP_VALUES = 16384;
    int                  index           = 0;
    std::vector<int>     completion_addrs;
};

enum CommunicationType
{
    ONE_SIDED = 1,
    TWO_SIDED = 2,
};

template <CommunicationType MODE, GPUMemoryType G>
class CXISend : public CXITrigger<G>, public CXIWait
{
    using FI_DFWQ_TYPE =
        std::conditional_t<MODE == CommunicationType::ONE_SIDED,
                           struct fi_op_rma, struct fi_op_msg>;

public:
    CXISend(Buffer local_completion, Buffer remote_completion,
            Buffer data_buffer, struct fid_domain* domain,
            struct fid_ep* main_ep, DeferredWorkQueue& dwq, fi_addr_t partner,
            fi_addr_t self)
        : CXIRequest(local_completion),
          CXITrigger<G>(domain),
          CXIWait(),
          domain_ptr(domain),
          my_queue(dwq),
          completion_a(CXICounter::alloc_counter(domain)),
          completion_b(CXICounter::alloc_counter(domain)),
          completion_c(CXICounter::alloc_counter(domain)),
          my_chained_completions(local_completion, remote_completion, main_ep,
                                 partner, self, completion_a, completion_b,
                                 completion_c)
    {
        // local_completion.print();
        // remote_completion.print();
        // data_buffer.print();

        work_entry          = {};
        message_description = {};
        msg_iov             = {};
        msg_rma_iov         = {};

        work_entry.threshold       = 0;
        work_entry.triggering_cntr = CXITrigger<G>::get_libfabric_counter();
        work_entry.completion_cntr = completion_a;
        if constexpr (CommunicationType::ONE_SIDED == MODE)
        {
            work_entry.op_type = FI_OP_WRITE;
        }
        else
        {
            work_entry.op_type = FI_OP_SEND;
            // TODO
            static_assert(false, "TWO SIDED not implemented yet");
        }
        work_entry.op.rma = &message_description;

        msg_iov     = {data_buffer.address, data_buffer.len};
        msg_rma_iov = {0, data_buffer.len, data_buffer.mr_key};

        message_description.ep            = main_ep;
        message_description.msg.msg_iov   = &msg_iov;
        message_description.msg.iov_count = 1;
        message_description.msg.addr      = partner;
        // No harm in doing this if mode is not one_sided
        message_description.msg.rma_iov       = &msg_rma_iov;
        message_description.msg.rma_iov_count = 1;
    }

    ~CXISend()
    {
        // Free counter
        force_libfabric(fi_close(&completion_a->fid));
        force_libfabric(fi_close(&completion_b->fid));
        force_libfabric(fi_close(&completion_c->fid));
    }

    void start_host() override
    {
        // Update threshold of chained things
        work_entry.threshold = threshold;
        // This one also set the "completion" value (address) to send
        my_chained_completions.set_active_threshold(threshold);

        // Queue up send of data
        my_queue.make_space(completion_c);
        force_libfabric(
            fi_control(&domain_ptr->fid, FI_QUEUE_WORK, &work_entry));

        // Queue up chained actions
        my_chained_completions.queue_work(domain_ptr);
        my_queue.consume();
    }

private:
    // Structs for the DFWQ Entry:
    struct fi_deferred_work work_entry;
    FI_DFWQ_TYPE            message_description;
    struct iovec            msg_iov;
    // Always here, if if not always used.
    struct fi_rma_iov msg_rma_iov;
    // Keep the domain pointer for later use
    struct fid_domain* domain_ptr;

    DeferredWorkQueue& my_queue;

    struct fid_cntr*  completion_a;
    struct fid_cntr*  completion_b;
    struct fid_cntr*  completion_c;
    ChainedRMA<false> my_chained_completions;
};

class CXIRecvOneSided : public CXIWait
{
public:
    CXIRecvOneSided(Buffer& comp_buffer, Buffer& data_buffer,
                    struct fid_domain* domain, struct fid_ep* main_ep)
        : CXIRequest(comp_buffer)
    {
        uint64_t recv_key_requested = getMRID();

        force_libfabric(fi_mr_reg(domain, data_buffer.address, data_buffer.len,
                                  FI_REMOTE_WRITE, 0, recv_key_requested,
                                  FI_MR_ALLOCATED, &my_mr, NULL));
        force_libfabric(fi_mr_bind(my_mr, &(main_ep)->fid, 0));

        // Enable MR
        force_libfabric(fi_mr_enable(my_mr));
        data_buffer.mr_key = fi_mr_key(my_mr);
    }

    ~CXIRecvOneSided()
    {
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
    }

    void start_host() override
    {
        // Do nothing
    }

    void start_gpu(hipStream_t* the_stream) override
    {
        // Do nothing
    }

private:
    // Allocated Libfabric Objects
    struct fid_mr* my_mr;
};

class CXIQueue : public Queue
{
public:
    using CXIObjects = std::unique_ptr<CXIRequest>;

    CXIQueue(hipStream_t* stream_addr)
        : comm_base(MPI_COMM_WORLD), the_stream(stream_addr)
    {
        int size;
        force_mpi(MPI_Comm_size(comm_base, &size));
        force_mpi(MPI_Comm_rank(comm_base, &my_rank));
        peers.resize(size, 0);
        libfabric_setup(size);
        peer_setup(size);
    }

    ~CXIQueue()
    {
        libfabric_teardown();
    }

    void enqueue_operation(std::shared_ptr<Request> qe) override
    {
        CXIObjects& cxi_stuff = request_map.at(qe->getID());
        cxi_stuff->start(the_stream);

        // Keep track of active requests
        active_requests.push_back(qe->getID());
    }

    void enqueue_waitall() override;

    void host_wait() override
    {
        force_hip(hipStreamSynchronize(*the_stream));
    }

    void match(std::shared_ptr<Request> qe) override
    {
        if (Communication::Operation::BARRIER == qe->operation)
        {
            /* Not really a true buffer! */
            Buffer blank(qe->buffer, 0, 0, 0);
            /* Add request to map */
            request_map.insert(std::make_pair(
                qe->getID(),
                std::make_unique<FakeBarrier>(blank, qe->comm, my_queue)));
        }
        else
        {
            prepare_cxi_mr_key(*qe);
        }
    }

private:
    void libfabric_setup(int size);
    void peer_setup(int rank);
    void prepare_cxi_mr_key(Request&);
    void libfabric_teardown();

    // Persistent Libfabric objects
    struct fi_info*    fi;       /*!< Provider's data and features */
    struct fid_fabric* fabric;   /*!< Represents the network */
    struct fid_domain* domain;   /*!< A subsection of the network */
    struct fid_av*     av;       /*!< Address vector for connections */
    struct fid_ep*     ep;       /*!< An endpoint */
    struct fid_cq*     txcq;     /*!< The transmit completion queue */
    struct fid_cq*     rxcq;     /*!< The receive completion queue */
    struct fid_cntr*   recv_ctr; /*!< The counters for receiving */

    // Peer information
    MPI_Comm               comm_base;
    int                    my_rank;
    std::vector<fi_addr_t> peers;

    // Map of Request ID to CXIObject (counters, mr)
    std::map<size_t, CXIObjects> request_map;
    std::vector<size_t>          active_requests;

    // Completion buffers
    static CompletionBuffer my_buffer;

    // Deferred Work Queue Management
    static DeferredWorkQueue my_queue;

    // Hip Stream
    hipStream_t* the_stream;
};

#endif
