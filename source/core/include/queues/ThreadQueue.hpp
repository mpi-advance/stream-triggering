#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <atomic>
#include <memory>

#include "abstract/queue.hpp"

/**
 * @file ThreadQueue.hpp
 * @brief Code for the implementation of a thread-based Queue backend.
 * @details See ::ThreadQueue class description for limitations of this backend.
 */

/**
 * @brief A Queue implementation that uses threads (and MPI) as its main backend.
 * @details This queue implementation is essentially just a progress thread that uses MPI
 * to perform the communication requests it receives. As such, this backend makes no
 * attempts to manually interact or align with any GPU operations the user may have
 * started. There is no way to give a GPU stream to this class so when using this
 * backend in real code, the user must still take care to synchronize with any GPU kernels
 * they are doing (if needed). See Progress::Engine for details on how the progress
 * thread spawns.
 * @sa Queue
 */
class ThreadQueue : public Queue
{
public:
    /** @brief Default constructor, only has a specific debug print. */
    ThreadQueue()
    {
        Print::out("Thread Queue init-ed");
    }
    /** @brief Default destructor. */
    ~ThreadQueue() = default;

    /** [UPDATE]
     * @brief Enqueue a specific communication to the ::ThreadQueue.
     * @details This function uses the id inside @p request (Communication::Request::myID)
     * as the index in the operation cache. If no entry exists for this ID yet, a new
     * ::ThreadQueueEntry is created and inserted into the map (Queue::request_cache).
     * After that check, the ::ThreadQueueEntry is retrieved from the map cache and
     * provided to the progress thread (Queue::progress_engine).
     *
     * Each time an operation is started, its threshold is incremented
     * (QueueEntry::increment), in theory aligning with the concept that the same
     * operation could be queued up multiple times. In practice, for this backend, that is
     * untested since this project focuses on GPU stream-triggered communication.
     *
     * The corresponding ::ThreadQueueEntry is also added to ::Queue::entries so that
     * this backend can keep track of all operations that have been started.
     *
     * Because the thread backend is a basic backend that does not tie to another stream,
     * once the ::ThreadQueueEntry has been enqueued, the trigger
     * (QueueEntry::start_location) for the start is immediately set to the correct value
     * (via QueueEntry::get_threshold()) for the start to be triggered.
     * @param [in] requests
     * @sa Communication::Request::getID()
     */
    void enqueue_startall(std::vector<std::shared_ptr<Request>> requests) override
    {
        Progress::CounterType target = build_start(requests);

        /* Basic thread implementation can instantly start request. */
        std::atomic_ref<Progress::CounterType>(*start_signal)
            .store(target, std::memory_order_release);
    }

    /** [UPDATE]
     * @brief Iterates over all started entries and creates a Progress::WaitEntry for
     * each one.
     * @details Note that each entry is currently enqueued separately and not all together
     * in a single "waitall" action. Queue::entries is cleared after this method is
     * called.
     * @param [in] requests
     * @param [out] statuses
     */
    void enqueue_waitall(std::vector<std::shared_ptr<Request>> requests,
                         MPI_Status*                           statuses) override
    {
        build_wait(requests, statuses);
    }
};

#endif