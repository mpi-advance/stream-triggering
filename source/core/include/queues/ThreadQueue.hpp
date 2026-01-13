#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <memory>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

class ThreadQueueEntry : public QueueEntry
{
public:
    ThreadQueueEntry(std::shared_ptr<Request> req) : QueueEntry(req)
    {
        start_location  = new Progress::CounterType();
        *start_location = 0;
        wait_location   = new Progress::CounterType();
        *wait_location  = 0;
        initialize_lambdas();
    }

    ~ThreadQueueEntry()
    {
        delete start_location;
        delete wait_location;
    }
};

class ThreadQueue : public Queue
{
public:
    ThreadQueue()
    {
        Print::out("Thread Queue init-ed");
    }
    ~ThreadQueue() = default;

    void enqueue_operation(std::shared_ptr<Request> request) override
    {
        size_t request_id = request->getID();
        if (!request_cache.contains(request_id))
        {
            // Also converts to InternalRequest
            request_cache.emplace(request_id,
                                  std::make_unique<ThreadQueueEntry>(request));
        }

        QueueEntry&           req       = *request_cache.at(request_id);
        Progress::CounterType threshold = req.increment();
        progress_engine.enqueued_start(req, threshold);
        entries.push_back(req);

        /* Basic thread implementation can instantly start request. */
        *(req.get_start_location()) = threshold;
    }

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