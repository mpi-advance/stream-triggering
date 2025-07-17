#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>

#include "abstract/bundle.hpp"
#include "abstract/entry.hpp"
#include "abstract/match.hpp"
#include "abstract/queue.hpp"

template <bool isSerialized>
class ThreadQueue : public Queue
{
public:
    using InternalRequest = QueueEntry;
    using UserRequest     = std::shared_ptr<Request>;

    ThreadQueue() : thr(&ThreadQueue::progress, this)
    {
        Print::out("Thread Queue init-ed");
    }
    ~ThreadQueue()
    {
        shutdown = true;
        thr.join();
    }

    void enqueue_operation(UserRequest request) override
    {
        size_t request_id = request->getID();
        if (!request_cache.contains(request_id))
        {
            // Also converts to InternalRequest
            request_cache.emplace(request_id, request);
        }
        entries.add_to_bundle(request_cache.at(request_id));
    }

    void enqueue_startall(std::vector<UserRequest> requests) override
    {
        for (auto& req : requests)
        {
            enqueue_operation(req);
        }
    }

    void enqueue_waitall() override
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        // Move Bundle (entries) to the queue of work
        pending.push(std::move(entries));
        // Add one to busy counter
        busy += 1;
        // Remake entries
        entries = Bundle();
    }

    void host_wait() override
    {
        while (busy.load())
        {
            // Do nothing.
        }
    }

protected:
    // Thread control variables
    std::atomic<int> busy;
    std::thread      thr;
    bool             shutdown = false;
    std::mutex       queue_guard;

    // Bundle variables
    using BundleIterator = std::vector<Bundle>::iterator;
    Bundle                            entries;
    std::queue<Bundle>                pending;
    std::map<size_t, InternalRequest> request_cache;

    void progress()
    {
        while (!shutdown)
        {
            if (busy > 0)
            {
                Bundle the_bundle = std::move(pending.front());
                {  // Scope of the lock
                    std::scoped_lock<std::mutex> incoming_lock(queue_guard);
                    pending.pop();
                }

                if constexpr (isSerialized)
                {
                    the_bundle.progress_serial();
                }
                else
                {
                    the_bundle.progress_all();
                }
                busy--;
            }
            else
            {
                std::this_thread::yield();
            }
        }
    }
};

#endif