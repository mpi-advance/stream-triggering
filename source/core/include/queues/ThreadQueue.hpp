#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <memory>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

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
            request_cache.emplace(request_id, request);
            request_cache.at(request_id).initialize_lambdas();
        }
        QueueEntry& req = request_cache.at(request_id);
        progress_engine.enqueued_start(req, req.increment());
        entries.push_back(req);

        /* Basic thread implementation can instantly start request. */
        *(req.get_start_location()) = 1;
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