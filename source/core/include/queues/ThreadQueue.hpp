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

    /** @brief Enqueue the starts of a set of operations to the ::ThreadQueue.
     * @details Effectively offloads the starting to Queue::build_start(). Since this
     * thread implementation is the "stream", it goes ahead and immediately marks the
     * triggering counter to the triggering threshold (after enqueueing the start of the
     * operations).
     * @param [in] requests The vector of Communication::Request objects to enqueue the
     * start of.
     * @sa Queue::build_start()
     */
    void enqueue_startall(std::vector<std::shared_ptr<Request>> requests) override
    {
        Progress::CounterType target = build_start(requests);

        /* Basic thread implementation can instantly start request. */
        std::atomic_ref<Progress::CounterType>(*start_signal)
            .store(target, std::memory_order_release);
    }

    /** @brief Enqueue the wait for a set of operations to the ::ThreadQueue.
     * @details Just calls out to Queue::build_wait().
     * @param [in, out] requests The Communication::Request operations to have their wait
     * enqueued.
     * @param [out] statuses A pointer to an array of MPI_Status objects to be filled in
     * upon the completion of the waitall. Pointer should be valid until the waitall is
     * done. Null pointer is mapped to MPI_STATUS_IGNORE.
     */
    void enqueue_waitall(std::vector<std::shared_ptr<Request>> requests,
                         MPI_Status*                           statuses) override
    {
        build_wait(requests, statuses);
    }
};

#endif