#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <memory>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

/**
 * @file ThreadQueue.hpp
 * @brief Code for the implementation of a thread-based Queue backend.
 * @details See ::ThreadQueue class description for limitations of this backend.
 */

/**
 * @brief ::QueueEntry derived class specific to the thread backend.
 * @details The only difference between this class and the base class is that it allocates
 * heap (host) memory for both ::QueueEntry::start_location and
 * ::QueueEntry::wait_location in the constructor, and frees them in the destructor.
 * Currently, this class does not have any need for the ::QueueEntry::wait_location, and
 * it is only created so that the base class functions can work correctly, allowing the
 * non-CXI backends to share common code.
 */
class ThreadQueueEntry : public QueueEntry
{
public:
    /**
     * @brief Basic constructor for ThreadQueueEntry.
     * @details Passes @p req onto the ::QueueEntry constructor and allocates
     * ::QueueEntry::start_location and ::QueueEntry::wait_location on the heap.
     * Sets the value at both of those locations to zero.
     * @param [in] req The Communication::Request this ::QueueEntry will be in charge of
     * creating start and wait actions for.
     */
    ThreadQueueEntry(std::shared_ptr<Request> req) : QueueEntry(req)
    {
        start_location  = new Progress::CounterType();
        *start_location = 0;
        wait_location   = new Progress::CounterType();
        *wait_location  = 0;
    }

    /** @brief Destructor that frees ::QueueEntry::start_location and
     * ::QueueEntry::wait_location. */
    ~ThreadQueueEntry()
    {
        delete start_location;
        delete wait_location;
    }
};

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

    /**
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
     * @param [in] request The Communication::Request corresponding to the communication
     * that should be enqueued.
     * @sa Communication::Request::getID()
     */
    void enqueue_operation(std::shared_ptr<Request> request) override
    {
        size_t request_id = request->getID();
        if (!request_cache.contains(request_id))
        {
            request_cache.emplace(request_id,
                                  std::make_unique<ThreadQueueEntry>(request));
        }

        QueueEntry& req = *request_cache.at(request_id);
        req.increment();
        progress_engine.enqueued_start(req);
        entries.push_back(req);

        /* Basic thread implementation can instantly start request. */
        *(req.get_start_location()) = req.get_threshold();
    }

    /** @brief Iterates over all started entries and creates a Progress::WaitEntry for
     * each one.
     * @details Note that each entry is currently enqueued separately and not all together
     * in a single "waitall" action. Queue::entries is cleared after this method is
     * called.
     */
    void enqueue_waitall() override
    {
        for (QueueEntry& entry : entries)
        {
            progress_engine.enqueued_wait(entry);
        }
        entries.clear();
    }
};

#endif