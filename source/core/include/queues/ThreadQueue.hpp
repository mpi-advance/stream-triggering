#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>

#include "abstract/queue.hpp"

class ThreadRequest
{
public:
    ThreadRequest(std::shared_ptr<Request> qe);

    void start();
    void prepare();
    bool canGo();
    bool done();

protected:
    static const int THREAD_PREPARE_TAG = 12999;

    MPI_Request            mpi_request;
    int                    prepare_request_buffer = -1;
    MPI_Request            prepare_request;
    std::weak_ptr<Request> original_request;
};

template <bool isSerialized>
class ThreadQueue : public Queue
{
public:
    ThreadQueue() : thr(&ThreadQueue::progress, this) {}
    ~ThreadQueue()
    {
        shutdown = true;
        thr.join();
    }

    void enqueue_operation(std::shared_ptr<Request> request) override
    {
        enqueue_aciton<Bundle::ThreadRequestAction::START>(request);
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
            // No lock because this function doesn't want to set
            // amount_to_do to any value, only read.
        }
    }

    void match(std::shared_ptr<Request> request) override
    {
        // Normal matching
        request->match();
        // But need to save request for later?
        auto match_info = request->getMatch();
        if (std::nullopt == match_info)
            throw std::runtime_error("Request was not matched properly!");
        remote_partner_to_local[std::make_tuple(request->peer, *match_info)] =
            request;
    }

protected:
    class Bundle
    {
    public:
        enum ThreadRequestAction
        {
            START   = 0,
            PREPARE = 1,
        };

        using RequestIterator = std::vector<
            std::tuple<ThreadRequestAction, ThreadRequest&>>::iterator;

        // No fancy constructor
        Bundle() {};

        void add_to_bundle(ThreadRequestAction operation,
                           ThreadRequest&      request)
        {
            items.emplace_back(operation, request);
        }

        void progress_serial()
        {
            // Start and progress operations one at a time
            for (RequestIterator entry = items.begin(); entry != items.end();
                 entry++)
            {
                ThreadRequestAction action = std::get<0>(*entry);
                ThreadRequest&      req    = std::get<1>(*entry);
                if (ThreadRequestAction::START == action)
                {
                    req.start();
                    while (!req.done())
                    {
                        // Do nothing
                    }
                }
                else if (ThreadRequestAction::PREPARE == action)
                {
                    req.prepare();
                }
            }
        }
        void progress_all()
        {
            // Start all actions
            for (RequestIterator entry = items.begin(); entry != items.end();
                 entry++)
            {
                ThreadRequestAction action = std::get<0>(*entry);
                ThreadRequest&      req    = std::get<1>(*entry);
                if (ThreadRequestAction::START == action)
                {
                    req.start();
                }
                else if (ThreadRequestAction::PREPARE == action)
                {
                    req.prepare();
                }
            }

            // Wait for "starts" to complete:
            for (RequestIterator entry = items.begin(); entry != items.end();
                 entry++)
            {
                ThreadRequestAction action = std::get<0>(*entry);
                ThreadRequest&      req    = std::get<1>(*entry);
                if (ThreadRequestAction::START == action)
                {
                    while (!req.done())
                    {
                        // Do nothing
                    }
                }
            }
        }

    private:
        std::vector<std::tuple<ThreadRequestAction, ThreadRequest&>> items;
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

    // Matching related variables
    using MessageID = std::tuple<int, int>;  //  <rank, request ID>
    std::map<MessageID, std::shared_ptr<Request>> remote_partner_to_local;

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

    template <Bundle::ThreadRequestAction TA>
    void enqueue_aciton(std::shared_ptr<Request> request)
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        size_t                       request_id = request->getID();
        if (!request_cache.contains(request_id))
        {
            request_cache.emplace(request_id, request);
        }
        entries.add_to_bundle(TA, request_cache.at(request_id));
    }
};

#endif