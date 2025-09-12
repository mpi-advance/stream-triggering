
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
#include "safety/hip.hpp"
#include "safety/libfabric.hpp"
#include "safety/mpi.hpp"

/** @brief calculate the memory size of the buffer in req
 * @param [in] req Request with the buffer to size
 * @return the memory size of the buffer 
 */
static inline MPI_Count get_size_of_buffer(Request& req)
{
    int size = -1;
    check_mpi(MPI_Type_size(req.datatype, &size));
    return size * req.count;
}

/** @brief Debug function prints out information about supplied deferred work queue entry
 *  @param [in] dfwq_entry deferred work quque entry
 *  @param [in] entry_name Name of the entry being printed. 
 */
static inline void print_entry(struct fi_deferred_work* dfwq_entry,
                                    std::string              entry_name)
{
    Print::out("--- Start:", entry_name);

    Print::out("Threshold:", dfwq_entry->threshold);
    Print::out("Local IOVEC:", dfwq_entry->op.rma->msg.msg_iov->iov_base,
               dfwq_entry->op.rma->msg.msg_iov->iov_len);
    Print::out("Remote IOVEC:", dfwq_entry->op.rma->msg.rma_iov->addr,
               dfwq_entry->op.rma->msg.rma_iov->len,
               dfwq_entry->op.rma->msg.rma_iov->key);

    Print::out("--- End", entry_name, "---");
}

/** @brief Wrapper class around metadata for a buffer. */
class Buffer
{
public:
    Buffer(void* _address, size_t _len, uint64_t _mr_key, uint64_t _offset)
        : address(_address), len(_len), mr_key(_mr_key), offset(_offset)
    {
    }

	/** @brief override of Libfabric remote memory access operator
	 * @return fi_rma_iov struct based on the information in the object. 
	 */
    operator fi_rma_iov() const
    {
        return {offset, len, mr_key};
    }

	/** @brief print out members of this object*/
    void print()
    {
        Print::out("Buffer:", address, len, mr_key, offset);
    }

	/** @brief local buffer address */
    void*    address;
	/** @brief size of target buffer */
    size_t   len;
	/** @brief access key */
    uint64_t mr_key;
	/** @brief target RMA address */
    uint64_t offset;
};

/** @brief class to control threshold trigger variable. 
 *  @details
 *      \todo what is difference between value and counter_value
 * 
 */
class Threshold
{
public:
    Threshold() : _value(0), _counter_value(0) {}

	/** @brief increment threshold counter */
    void increment_threshold()
    {
        _value++;
    }

	/** @brief calculate the difference between value and counter value and reset counter_value. 
	 *  @details
	 *			resets value to counter_value
	 *
	 *  @return the difference between value and counter value. 
	 */
    size_t equalize_counter()
    {
        size_t diff    = _value - _counter_value;
        _counter_value = _value;
        return diff;
    }

    /** @brief getter for Threshold._value */
    size_t value()
    {
        return _value;
    }
	
	 /** @brief getter for Threshold._counter_value */
    size_t counter_value()
    {
        return _counter_value;
    }

private:
	/** @brief  */ 
    size_t _value         = 0;
	/** @brief  */
    size_t _counter_value = 0;
};

/** @brief abstraction on top of CXI libfabric DefferedWorkQueue
 *  @details
 *     Manages entries in deferred workqueue
 * 
 */
class DeferredWorkQueue
{
public:
    DeferredWorkQueue() = default;
	/** @brief store pointer to progress counter*/
    void register_progress_counter(struct fid_cntr* p_cntr)
    {
        Print::out("Currently in use:", space_used);
        progress_cntr = p_cntr;
    }

	/** @brief add completion counter to list*/
    void register_completion_counter(struct fid_cntr* c_cntr)
    {
        if (!known_completion_map.contains(c_cntr))
        {
            Print::out("Adding completion counter:", c_cntr);
            known_completion_map.insert({c_cntr, 0});
        }
    }

	/** @brief increments space_used when ????*/
    void consume()
    {
        Print::out("*** Consumed");
        space_used++;
    }

	/** @brief Remove a completion counter from tracking
	 * @details
	 *  delete completion counter and mark space used as freed. 
	 *  starts new requests until max_threshold is reached
	 * 
	 * @param [in, out] completion_cntr completion counter
	 * @param [in] max_threshold
	 */
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



	/** @brief
	 * @details
	 *  
	 * @param [in, out] completion_cntr completion counter
	 */
    void make_space(struct fid_cntr* completion_cntr)
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
	
	/** @brief check on progress counter*/
    uint64_t progress()
    {
        return fi_cntr_read(progress_cntr);
    }

	/** @brief check on progress counter*/
    void print_status()
    {
        Print::always("Space in use:", space_used);
    }

private:
    //goes through known counters and updates free space for more operations in dwq. 
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
    const uint64_t total_space = 84; //unit is CXISends which consist of 3 libfabric operations. 
    uint64_t       space_used  = 0;

    /** @brief  Progress counter **/
    struct fid_cntr* progress_cntr;

    // Last known completions
	/** @brief signals to host that request is complete on gpu **/
    std::map<struct fid_cntr*, uint64_t> known_completion_map;
};

/** @brief 
 *  @details
 *
 * 
 */
class CXICounter
{
public:
	//wrapper for libF counter
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

	//wrapper for cxi extension and libf counter 
    CXICounter(struct fid_domain* domain) : counter(alloc_counter(domain))
    {
        // Open (create) CXI Extension object
        check_libfabric(fi_open_ops(&(counter->fid), FI_CXI_COUNTER_OPS, 0,
                                    (void**)&counter_ops, NULL));
        // Get the MMIO Address of the counter
        check_libfabric(
            counter_ops->get_mmio_addr(&counter->fid, &mmio_addr, &mmio_addr_len));
        // Register MMIO Address w/ HIP
        force_hip(hipHostRegister(mmio_addr, mmio_addr_len, hipHostRegisterDefault));
        // Get GPU version of MMIO address
        force_hip(hipHostGetDevicePointer(&gpu_mmio_addr, mmio_addr, 0));
    }

    ~CXICounter()
    {
        // Free counter
        force_libfabric(fi_close(&counter->fid));
        force_hip(hipHostUnregister(mmio_addr));
    }

	//print out counter value
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

/** @brief Get unique key id for memory location to pass between peer processes */
// MR Management
static size_t getMRID()
{
    static size_t ID = 1;
    return ID++;
}

/** @brief abstraction around concept of completion buffer
 *  @details
 *    array of memory slots for completions
 * 
 */
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

	//registers completion buffers inside libFabric space. 
    void register_mr(struct fid_domain* domain, struct fid_ep* main_ep)
    {
        force_libfabric(fi_mr_reg(domain, buffer, DEFAULT_SIZE,
                                  FI_REMOTE_WRITE | FI_WRITE, 0, getMRID(),
                                  FI_MR_ALLOCATED, &my_mr, NULL));
        force_libfabric(fi_mr_bind(my_mr, &(main_ep)->fid, 0));

        // Enable MR
        force_libfabric(fi_mr_enable(my_mr));
    }

	// necessary for order of operations during cleanup (statics get killed last, needs to cleanup sooner)
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
        void*    x            = ((char*)buffer) + (sizeof(size_t) * current_index);
        uint64_t offset_value = current_index * DEFAULT_ITEM_SIZE;
        current_index++;
        return Buffer(x, DEFAULT_SIZE, fi_mr_key(my_mr), offset_value);
    }

    struct fid_mr* my_mr;
    void*          buffer;
    size_t         current_index;

    static constexpr size_t DEFAULT_ITEMS     = 1000; //default limit for number of completions
    static constexpr size_t DEFAULT_ITEM_SIZE = sizeof(size_t); //size of "completion"
    static constexpr size_t DEFAULT_SIZE      = DEFAULT_ITEMS * DEFAULT_ITEM_SIZE; //"amount of memory to allocate to hold completions"
};

/** @brief 
 *  @details
 *
 * 
 */
class CXIRequest
{
public:
    CXIRequest(GPUMemoryType _memory_type, Buffer local_completion_buffer)
        : memory_type(_memory_type),
          completion_buffer(local_completion_buffer),
          num_times_started(0)
    {
    }
    virtual ~CXIRequest() = default;
	
	/** @brief */
    virtual void start(hipStream_t* the_stream, Threshold& threshold,
                       CXICounter& trigger_cntr)
    {
        num_times_started++;
        start_host(the_stream, threshold, trigger_cntr);
        start_gpu(the_stream, threshold, trigger_cntr);
    }

	/** @brief enqueue kernel to gpu and wait for request completion */
    virtual void wait_gpu(hipStream_t* the_stream) = 0;

	/** @brief getter for GPUMemoryType **/
    virtual GPUMemoryType get_gpu_memory_type()
    {
        return memory_type;
    }

protected:
    virtual void start_host(hipStream_t* the_stream, Threshold& threshold,
                            CXICounter& trigger_cntr) = 0;
    virtual void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                           CXICounter& trigger_cntr)  = 0;

    GPUMemoryType memory_type;
    Buffer        completion_buffer;
    size_t        num_times_started;
};

/** @brief 
 *  @details
 *
 * 
 */
class FakeBarrier : public CXIRequest
{
public:
    // Uses "GPUMemoryType::FINE" because this doesn't need a flush
    FakeBarrier(Buffer buffer, MPI_Comm comm, DeferredWorkQueue& dwq)
        : CXIRequest(GPUMemoryType::FINE, buffer),
          comm_to_use(comm),
          finished(true),
          progress_engine(dwq)
    {
        // Setup GPU memory locations
        force_hip(hipHostMalloc((void**)&host_start_location, sizeof(int64_t),
                                hipHostMallocDefault));
        *host_start_location = 0;
        force_hip(hipHostGetDevicePointer(&gpu_start_location, host_start_location, 0));
        force_hip(hipHostMalloc((void**)&host_wait_location, sizeof(int64_t),
                                hipHostMallocDefault));
        *host_wait_location = 0;
        force_hip(hipHostGetDevicePointer(&gpu_wait_location, host_wait_location, 0));
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
        force_hip(
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
        MPI_Barrier(comm_to_use);

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
    // MPI Variables
    MPI_Comm comm_to_use;
    // Thread
    std::thread thr;
    bool        finished = true;
    // Progress
    DeferredWorkQueue& progress_engine;
};

/** @brief wait for the stream to complete**/
class CXIWait : virtual public CXIRequest
{
public:
    void wait_gpu(hipStream_t* the_stream) override;
};

/** @brief 
 *  @details
 *
 * 
 */
template <bool FENCE = false>
class ChainedRMA
{
public:
    ChainedRMA(Buffer local_completion, struct fid_ep* ep, fi_addr_t partner,
               fi_addr_t self, struct fid_cntr* trigger, struct fid_cntr* remote_cntr,
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
        /* Filled in by the type cast of Buffer class */
        local_rma_iov = local_completion;
        /* The first and last values should be filled in by the match! */
        remote_rma_iov = {0, CompletionBuffer::DEFAULT_ITEM_SIZE, 0};

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
    void queue_work(struct fid_domain* domain)
    {
        /* Increase thresholds before starting! */
        chain_work_remote.threshold++;
        chain_work_local.threshold++;
        chain_iovec = {&completion_addrs.at(index++), sizeof(int)};

        // print_dfwq_entry(&chain_work_remote, "Chain work remote completion");
        check_libfabric(fi_control(&domain->fid, FI_QUEUE_WORK, &chain_work_remote));
        // print_dfwq_entry(&chain_work_local, "Chain work local completion");
        check_libfabric(fi_control(&domain->fid, FI_QUEUE_WORK, &chain_work_local));
    }

    uint64_t* get_rma_iov_addr_addr()
    {
        return &(remote_rma_iov.addr);
    }

    uint64_t* get_rma_iov_key_addr()
    {
        return &(remote_rma_iov.key);
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

    static constexpr int MAX_COMP_VALUES = 100000;
    int                  index           = 0;
    std::vector<int>     completion_addrs;
};

/** @brief Communication model the request will use. 
 *  @details 
 *  While TWO_SIDED is a valid value the functionality is not currently supported by the rest of the
 *  library. 
 * 
 */ 
enum CommunicationType
{
    ONE_SIDED = 1, //!< One-sided communication (WRITE/READ) will used used 
    TWO_SIDED = 2, //!< Two-sided communication (Send/Recv) not yet implemented
};

/** @brief 
 *  @details
 *    Operations used depend on MODE of communication selected. 
 * 
 */
template <CommunicationType MODE>
class CXISend : public CXIWait
{
    using FI_DFWQ_TYPE = std::conditional_t<MODE == CommunicationType::ONE_SIDED,
                                            struct fi_op_rma, struct fi_op_msg>;

public:
	/** @brief 
	*  @details
	*     
	* 
	*/
    CXISend(Buffer local_completion, Request& user_request, struct fid_domain* domain,
            struct fid_ep* main_ep, DeferredWorkQueue& dwq, fi_addr_t partner,
            fi_addr_t self)
        : CXIRequest(user_request.get_memory_type(), local_completion),
          CXIWait(),
          domain_ptr(domain),
          my_queue(dwq),
          completion_a(CXICounter::alloc_counter(domain)),
          completion_b(CXICounter::alloc_counter(domain)),
          completion_c(CXICounter::alloc_counter(domain)),
          my_chained_completions(local_completion, main_ep, partner, self, completion_a,
                                 completion_b, completion_c)
    {
        my_queue.register_completion_counter(completion_c);
        work_entry          = {};
        message_description = {};
        msg_iov             = {};
        msg_rma_iov         = {};

        work_entry.threshold = 0;
        /* Wait to set "work_entry.triggering_cntr" when request is enqueued. */

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

        auto data_len = static_cast<size_t>(get_size_of_buffer(user_request));
        msg_iov       = {user_request.buffer, data_len};
        /* Last field will be filled in by the match! */
        msg_rma_iov = {0, data_len, 0};

        message_description.ep            = main_ep;
        message_description.msg.msg_iov   = &msg_iov;
        message_description.msg.iov_count = 1;
        message_description.msg.addr      = partner;
        // No harm in doing this if mode is not one_sided
        message_description.msg.rma_iov       = &msg_rma_iov;
        message_description.msg.rma_iov_count = 1;

        /* Start requests to get data from peer */
        std::vector<uint64_t*> matching_data(3);
        matching_data.at(0) = &msg_rma_iov.key;
        matching_data.at(1) = my_chained_completions.get_rma_iov_key_addr();
        matching_data.at(2) = my_chained_completions.get_rma_iov_addr_addr();
        Communication::OneSideMatch::take(matching_data, user_request);
    }


    /** @brief 
	*  @details
	*     
	* 
	*/
    ~CXISend()
    {
        my_queue.clear_completion_counter(completion_c,
                                          my_chained_completions.get_threshold());
        // Free counter
        force_libfabric(fi_close(&completion_a->fid));
        force_libfabric(fi_close(&completion_b->fid));
        force_libfabric(fi_close(&completion_c->fid));
    }

	/** @brief 
	*  @details
	*     
	* 
	*/
    void start_host(hipStream_t* the_stream, Threshold& threshold,
                    CXICounter& trigger_cntr) override
    {
        // Update threshold of chained things
        work_entry.threshold = threshold.value();
        // Adjust the triggering counter to use
        work_entry.triggering_cntr = trigger_cntr.counter;

        // Make sure we can add at 1 iteration (3 DFWQ entries technically)
        my_queue.make_space(completion_c);

        // print_dfwq_entry(&work_entry, "Send");
        try
        {
            // Queue up send of data
            force_libfabric(fi_control(&domain_ptr->fid, FI_QUEUE_WORK, &work_entry));
        }
        catch (std::exception& e)
        {
            my_queue.print_status();
            throw e;
        }

        // Queue up chained actions
        my_chained_completions.queue_work(domain_ptr);
        my_queue.consume();
    }

	/** @brief 
	*  @details
	*     
	* 
	*/
    void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                   CXICounter& trigger_cntr) override;

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

/** @brief 
 *  @details
 *
 * 
 */
class CXIRecvOneSided : public CXIWait
{
public:
    CXIRecvOneSided(Buffer& comp_buffer, Request& user_request, struct fid_domain* domain,
                    struct fid_ep* main_ep)
        : CXIRequest(user_request.get_memory_type(), comp_buffer)
    {
        uint64_t recv_key_requested = getMRID();

        force_libfabric(fi_mr_reg(domain, user_request.buffer,
                                  get_size_of_buffer(user_request), FI_REMOTE_WRITE, 0,
                                  recv_key_requested, FI_MR_ALLOCATED, &my_mr, NULL));
        force_libfabric(fi_mr_bind(my_mr, &(main_ep)->fid, 0));

        // Enable MR
        force_libfabric(fi_mr_enable(my_mr));
        mr_key_storage = fi_mr_key(my_mr);

        // Start match process
        std::vector<uint64_t*> matching_data(3);
        matching_data.at(0) = &mr_key_storage;
        matching_data.at(1) = &completion_buffer.mr_key;
        matching_data.at(2) = &completion_buffer.offset;
        Communication::OneSideMatch::give(matching_data, user_request);
    }

    ~CXIRecvOneSided()
    {
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
    }

    void start_host(hipStream_t* the_stream, Threshold& threshold,
                    CXICounter& trigger_cntr) override
    {
        // Do nothing
    }

    void start_gpu(hipStream_t* the_stream, Threshold& threshold,
                   CXICounter& trigger_cntr) override
    {
        // Do nothing
    }

private:
    // Allocated Libfabric Objects
    struct fid_mr* my_mr;
    uint64_t       mr_key_storage;
};

/** @brief 
 *  @details
 *
 * 
 */
class CXIQueue : public Queue
{
public:
    using CXIObjects = std::unique_ptr<CXIRequest>;

    CXIQueue(hipStream_t* stream_addr)
        : comm_base(MPI_COMM_WORLD), the_stream(stream_addr)
    {
        Print::out("CXI Queue init-ed");
        int size;
        force_mpi(MPI_Comm_size(comm_base, &size));
        force_mpi(MPI_Comm_rank(comm_base, &my_rank));
        peers.resize(size, 0);
        libfabric_setup(size);
        peer_setup(size);
        the_gpu_counter = std::make_unique<CXICounter>(domain);
    }

    ~CXIQueue()
    {
        MPI_Barrier(comm_base);
        libfabric_teardown();
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

	/** @brief force synchronization with stream*/
    void host_wait() override
    {
        Print::out("Waiting on device!");
        force_hip(hipStreamSynchronize(*the_stream));
    }

	/** @brief match 
	 * @details
	 * Override of match function. 
	 * If Barrier Operation: 
	 * If not 
	*/
    void match(std::shared_ptr<Request> qe) override
    {
        if (Communication::Operation::BARRIER == qe->operation)
        {
            /* Not really a true buffer! */
            Buffer blank(qe->buffer, 0, 0, 0);
            /* Add request to map */
            request_map.insert(std::make_pair(
                qe->getID(), std::make_unique<FakeBarrier>(blank, qe->comm, my_queue)));
        }
        else
        {
            prepare_cxi_mr_key(*qe);
        }
    }

private:

    /** @brief 
	 *  @param [in] size
	 */
    void libfabric_setup(int size);
    
    /** @brief 
	 *  @param [in] size
	 */
	void peer_setup(int size);
	
	 /** @brief allocate local completion buffer
	 * @details
	 * Data to be exchanged (receiver sends this, sender receives this)
     * 1. MR key for receive buffer
     * 2. Remote completion buffer mr key
     * 3. Remote completion buffer offset  
	 * Inserts the used requests into the map. 
	 * 
	 *  @param [in] req
	 */
    void prepare_cxi_mr_key(Request&);
	
    /** @brief cleanup Libfabric framework, delete counters, buffers, and close other fi objects*/
    void libfabric_teardown();
    
	/** @brief load kernel that forces a write back of the L2 cache. 
	 *  @param [in,out] the_stream hip stream to force write back.  
	 */
	void flush_memory(hipStream_t*);

	/** @brief Start processing the stream. 
	 *  @param [in,out] req 
	 */
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
	/** @brief Provider's data and features */
    struct fi_info*    fi;     
    /** @brief Represents the network */
	struct fid_fabric* fabric;   
    /** @brief A subsection of the network */
	struct fid_domain* domain;   
    /** @brief Address vector for connections */
	struct fid_av*     av;       
    /** @brief An endpoint */
	struct fid_ep*     ep;       
    /** @brief The transmit completion queue */
	struct fid_cq*     txcq;     
    /** @brief The receive completion queue */
	struct fid_cq*     rxcq;     
    /** @brief The counters for receiving */
	struct fid_cntr*   recv_ctr; 

    /** @brief Communicator for context */
    MPI_Comm               comm_base;
    /** @brief rank of this process in the Communicator */
	int                    my_rank;
	/** @brief List of peers to communicate with */
    std::vector<fi_addr_t> peers;

    /** @brief Map of Request ID to CXIObject (counters, mr) */
    std::map<size_t, CXIObjects> request_map;
	
	/** @brief list of ids of active requests */
    std::vector<size_t>          active_requests;

    /** @brief buffer of memory slots to hold completion flags */
    static CompletionBuffer my_buffer;

    /** @brief Deferred Work Queue handle */
    static DeferredWorkQueue my_queue;

    /** @brief Hip Stream */
    hipStream_t* the_stream;
	
    /** @brief GPU Triggerable Counter */
    std::unique_ptr<CXICounter> the_gpu_counter;
	
	/** @brief memory locations for signaling between hosts */
    Threshold                   queue_thresholds;
};

#endif
