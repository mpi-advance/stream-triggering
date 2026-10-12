#ifndef ST_CXI_QUEUE_REQUEST_BASE
#define ST_CXI_QUEUE_REQUEST_BASE

#include "abstract/request.hpp"
#include "queues/cxi/completion_buffers.hpp"

/**
 * @file request_base.hpp
 * @brief File for the base ::CXIRequest and related enums and variables
 */

/** @brief Triggering action a given ::CXIRequest needs to happen.
 * @details Currently only one option is allowed to be chosen for a given operation.
 */
enum class TriggerStatus
{
    /** @brief No triggering actions are needed for this operation */
    NOT_NEEDED = 0,
    /** @brief The operation needs to be triggered by an increase of the global counter.
     * @sa CXIQueue::the_gpu_counter
     */
    GLOBAL_BUMP = 1,
    /** @brief The operation has its own counter that needs to be triggered. */
    EXTRA_BUMP = 2,
    /** @brief The operation does not need a counter trigger, and does not need to be
     * added to the list of "started" operations.
     * @sa CXIQueue::active_requests
     */
    DONE = 3,
};

/** @brief How an operation wants to enqueue its "wait for completion". */
enum class WaitStatus
{
    /** @brief The operation has enqueued its own kernel for waiting. */
    NOT_NEEDED = 0,
    /** @brief The operation would like to be added to the "mega" wait kernel. */
    GLOBAL_KERNEL = 1,
};

/** @brief Description of a single operation's completion. */
struct GPUCompletionDescription
{
    /** @brief The location of the completion buffer. */
    volatile CompletionBufferFactory::COMPLETION_TYPE* comp_addr;
    /** @brief The value that means "completed".*/
    size_t goal_value;
};

/** @brief A set of completion objects to be put in a kernel's parameter space.
 * @details This object is done this way to avoid an explicit call to copy the object into
 * GPU memory.
 * @sa ::wait_on_completions()
 */
struct GPUCompletions
{
    /** @brief The array of completions a kernel will work on.
     * @details Currently maxes out at CompletionBufferFactory::MAX_GPU_COMPLETIONS slots.
     */
    GPUCompletionDescription buffer[CompletionBufferFactory::MAX_GPU_COMPLETIONS];
    /** @brief How many slots are currently filled with valid info. */
    size_t actual_size;
};

/** @brief Helper alias to match ::GPUCompletionDescription for triggers. */
using GPUTriggerDescription = uint64_t*;
/** @brief A set of triggering locations to be put in a kernel's parameter space.
 * @details This object is done this way to avoid an explicit call to copy the object into
 * GPU memory. Does not support specifying a triggered increase greater than 1.
 * @sa ::add_to_counters()
 */
struct GPUTriggers
{
    /** @brief The array of counter addresses a kernel will trigger.
     * @details Currently maxes out at CompletionBufferFactory::MAX_GPU_COMPLETIONS slots.
     */
    GPUTriggerDescription buffer[CompletionBufferFactory::MAX_GPU_COMPLETIONS];
    /** @brief How many slots are currently filled with valid info. */
    size_t actual_size;
};

/**
 * @brief The abstract base class for the CXIQueue to describe operations with.
 * @details The base class attempts to describe, generally, all of the basic operations an
 * operation may need to do. The original concept was that there were four phases: 1)
 * start-cpu, 2) start-gpu, 3) wait-cpu, 4) wait-gpu. However, 2 ended up just being a
 * single step of 1 (launching the kernel), and 3 was more of the responsibility of the
 * object above this class (::CXIQueue).
 *
 * This class is also responsible for providing a match method, so derived classes must
 * decide how they want to match, based on the protocol they are supporting.
 *
 */
class CXIRequest
{
public:
    /**
     * @brief Base constructor. Stores the passed operation and allocates a completion
     * buffer.
     * @param [in] req The Communication::Request operation this class is tasked with
     * carrying out.
     * @param [in, out] buffers The buffer factory to allocate a completion buffer from.
     */
    CXIRequest(Communication::Request& req, CompletionBufferFactory& buffers)
        : base_req(req), completion_buffer(buffers.alloc_buffer()), num_times_started(0)
    {
    }

    /** @brief Default destructor. */
    virtual ~CXIRequest() = default;

    /**
     * @brief Carry out CPU actions related to starting the operation.
     * @details The base function automatically increases the counter for the number of
     * times started. If derived classes override this function, they must increment
     * ::CXIRequest::num_times_started on their own, or have a different mechanism for
     * determining the triggering and completion thresholds.
     *
     * Base function then calls ::CXIRequest::start_derived() to start GPU actions. Note
     * that method is not implemented by this class, so derived classes MUST implement it.
     *
     * Note that the @p trigger_cntr is the ::CXIQueue::the_gpu_counter that can be used
     * if the derived class does not need to use its own counter. Derived classes should
     * take care to return the appropriate ::TriggerStatus based on what they use.
     *
     * Derived classes may, but are not required to, enqueue operations to the provided
     * GPU stream.
     *
     * @param [in] trigger_cntr The global hardware counter that can be used.
     * @param [in] the_stream The GPU stream that should be enqueued to.
     * @return ::TriggerStatus indicating how the operation will be triggered (it may not
     * need to be).
     */
    virtual TriggerStatus start_cpu(CXICounter& trigger_cntr, hipStream_t* the_stream)
    {
        num_times_started++;
        return start_derived(trigger_cntr, the_stream);
    }

    /**
     * @brief Perform GPU actions related to waiting on the operation.
     * @details The base version does not do anything but be asked to be in the global
     * wait kernel.
     * @param [in] the_stream The GPU stream that should be enqueued to.
     * @return If this operation can be waited on in the global wait kernel. Base class
     * version returns ::WaitStatus::GLOBAL_KERNEL.
     */
    virtual WaitStatus wait_gpu(hipStream_t* the_stream);

    /** @brief Ask the underlying ::Communication::Request what its
     * ::Communication::GPUMemoryType is.
     * @return The ::Communication::GPUMemoryType of the underlying operation.
     * @sa ::Communication::Request::get_memory_type()
     */
    virtual Communication::GPUMemoryType get_gpu_memory_type()
    {
        return base_req.get_memory_type();
    }

    /**
     * @brief Initiate matching between two communications.
     * @details At the point this is called, the protocol has been decided, and should be
     * the same on both pairs of the communication (and are already technically matched
     * up). This method finishes the necessary exchanges in order to support the desired
     * ::Communication::Protocol.
     * @param [in] phase_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @param [in] phase_b Communication::MPIS_COMM_WORLD duplicate for Phase B
     * communication.
     * @sa ::Match
     */
    virtual void match(MPI_Comm phase_a, MPI_Comm phase_b) = 0;

    /** @brief Generate a ::GPUCompletionDescription for the operation.
     * @return A ::GPUCompletionDescription based on the given completion buffer's address
     * and the number of times this operation has been started.
     */
    GPUCompletionDescription get_gpu_completion()
    {
        return {(size_t*)completion_buffer.address, num_times_started};
    }

    /** @brief Generate a ::GPUTriggerDescription for the operation.
     * @details Note that not all classes will have a trigger description they can return.
     * It is up to the caller to check the return of CXIRequest::start_cpu(). The base
     * version will throw an exception if called.
     * @return A ::GPUTriggerDescription on how to trigger the start of the operation
     * (throws runtime error in base version).
     */
    virtual GPUTriggerDescription get_gpu_trigger()
    {
        throw std::runtime_error("Request was unable to offer trigger description");
    }

protected:
    /**
     * @brief Protocol-specific start actions (e.g. queueing triggered NIC operations, or
     * enqueuing stream work).
     * @details Unimplemented in the base class, required to be implemented by the child
     * class. See ::CXIRequest::start_cpu() on more details about the provided parameters.
     * @param [in] trigger_cntr The global hardware counter that can be used.
     * @param [in] the_stream The GPU stream that should be enqueued to.
     * @return ::TriggerStatus indicating how the operation will be triggered (it may not
     * need to be).
     */
    virtual TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                        hipStream_t* the_stream) = 0;

    /** @brief The underlying ::Communication::Request that describes the operation. */
    Communication::Request& base_req;
    /** @brief The completion buffer of the operation. */
    CompletionBuffer completion_buffer;
    /** @brief A buffer location to use for exchanging protocols during matching. */
    CompletionBuffer protocol_buffer;

    /** @brief Counter for how many times this operation has been started. */
    size_t num_times_started;
};

#endif