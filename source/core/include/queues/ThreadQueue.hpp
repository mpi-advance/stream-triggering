#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>

#include "abstract/match.hpp"
#include "abstract/queue.hpp"
#include "abstract/entry.hpp"

template <bool isSerialized>
class ThreadQueue : public Queue
{
public:
    using ThreadRequest = QueueEntry;
    ThreadQueue() : thr(&ThreadQueue::progress, this) {}
    ~ThreadQueue()
    {
        shutdown = true;
        thr.join();
    }

    void enqueue_operation(std::shared_ptr<Request> request) override
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        size_t                       request_id = request->getID();
        if (!request_cache.contains(request_id))
        {
            request_cache.emplace(request_id, request);
        }
        entries.add_to_bundle(request_cache.at(request_id));
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

    void match(std::shared_ptr<Request> request) override
    {
        // Normal matching
        Communication::BlankMatch();
        request->toggle_match();
    }

protected:
    class Bundle
    {
    public:
        using RequestIterator = std::vector<ThreadRequest>::iterator;

        // No fancy constructor
        Bundle() {};

        void add_to_bundle(ThreadRequest request)
        {
            items.emplace_back(request);
        }

        void progress_serial()
        {
            // Start and progress operations one at a time
            for (RequestIterator entry = items.begin(); entry != items.end();
                 entry++)
            {
                ThreadRequest& req = *entry;
                req.start();
                while (!req.done())
                {
                    // Do nothing
                }
            }
        }
        void progress_all()
        {
            // Start all actions
            for (RequestIterator entry = items.begin(); entry != items.end();
                 entry++)
            {
                ThreadRequest& req = *entry;
                req.start();
            }

            // Wait for "starts" to complete:
            for (RequestIterator entry = items.begin(); entry != items.end();
                 entry++)
            {
                ThreadRequest& req = *entry;
                while (!req.done())
                {
                    // Do nothing
                }
            }
        }

    private:
        std::vector<ThreadRequest> items;
    };

    // Thread control variables
    std::atomic<int> busy;
    std::thread      thr;
    bool             shutdown = false;
    std::mutex       queue_guard;

    // Bundle variables
    using BundleIterator = std::vector<Bundle>::iterator;
    Bundle                          entries;
    std::queue<Bundle>              pending;
    std::map<size_t, ThreadRequest> request_cache;

    void progress()
    {
        while (!shutdown)
        {
            if (pending.size() > 0)
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