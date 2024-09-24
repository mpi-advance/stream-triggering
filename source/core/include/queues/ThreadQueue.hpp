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
    ThreadRequest(Communication::Request* req);

    void start();
    bool canGo();
    bool done();

protected:
    MPI_Request mpi_request;
    Communication::Request* original_request;
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

    void enqueue_operation(Communication::Request* qe) override
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        entries.push_back(ThreadRequest(qe));
    }

    void enqueue_waitall() override
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        size_t                       amt = entries.size();
        stop_counts.push(amt);
        pending.insert(pending.begin(), entries.begin(), entries.end());
        busy += amt;
        entries.clear();
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

    void match(Communication::Request* request) override
    {
        // Normal matching
        request->match();
        // But need to save request for later?
        auto match_info = request->getMatch();
        if (std::nullopt == match_info)
            throw std::runtime_error("Request was not matched properly!");
        remote_partner_to_local[std::make_tuple(request->peer, *match_info)] = request;
    }

    void prepare(Communication::Request* request) override
    {
        if (Communication::Operation::RECV == request->operation)
        {
            int partner_id = request->getID();
            // TODO: Not use MPI_COMM_WORLD (in Recv mostly)
            force_mpi(MPI_Send(&partner_id, 1, MPI_INT, request->peer,
                               THREAD_PREPARE_TAG, MPI_COMM_WORLD));
            request->ready();
        }
    }

protected:
    static const int THREAD_PREPARE_TAG = 12999;

    std::atomic<int> busy;
    std::thread      thr;
    bool             shutdown = false;

    std::mutex queue_guard;

    using RequestIterator = std::vector<ThreadRequest>::iterator;
    std::vector<ThreadRequest> entries;
    std::vector<ThreadRequest> pending;
    std::vector<ThreadRequest> ongoing;
    std::queue<size_t>         stop_counts;

    // <rank, request ID>
    using MessageID = std::tuple<int, int>;
    std::map<MessageID, Communication::Request*> remote_partner_to_local;

    void progress()
    {
        // Thread specific variables
        size_t amount_to_do = 0;

        while (!shutdown)
        {
            if (amount_to_do == 0 && stop_counts.size() > 0)
            {
                // Determine how much we need to do
                {  // Scope of the lock
                    std::scoped_lock<std::mutex> incoming_lock(queue_guard);
                    amount_to_do = stop_counts.front();
                    stop_counts.pop();
                    ongoing.insert(ongoing.begin(), pending.begin(),
                                   pending.begin() + amount_to_do);
                    pending.erase(pending.begin(),
                                  pending.begin() + amount_to_do);
                }

                if constexpr (isSerialized)
                {
                    progress_options_serial(amount_to_do);
                }
                else
                {
                    progress_options_all(amount_to_do);
                }
            }
            else
            {
                //progress_prepares();
                std::this_thread::yield();
            }
        }
    }

    void progress_options_all(size_t& amount_to_do)
    {
        // Start operations
        for (RequestIterator entry = ongoing.begin(); entry != ongoing.end();
             entry++)
        {
            while(!(*entry).canGo())
            {
                progress_prepares();
            }
            (*entry).start();
        }

        // Progress them and watch out for shutdown (just in case)
        while (amount_to_do != 0 && !shutdown)
        {
            for (RequestIterator entry = ongoing.begin();
                 entry != ongoing.end(); entry++)
            {
                if ((*entry).done())
                {
                    ongoing.erase(entry);
                    amount_to_do--;
                    busy--;
                    break;
                }
                else
                {
                    progress_prepares();
                }
            }
        }
    }

    void progress_options_serial(size_t& amount_to_do)
    {
        // Start and progress operations one at a time
        for (RequestIterator entry = ongoing.begin(); entry != ongoing.end();
             entry++)
        {
            while(!(*entry).canGo())
            {
                progress_prepares();
            }
            (*entry).start();
            while (!(*entry).done())
            {
                progress_prepares();
            }
            amount_to_do--;
            busy--;
        }
        ongoing.clear();
    }

    void progress_prepares()
    {
        MPI_Status incoming_message;
        int        isThere = 0;
        force_mpi(MPI_Iprobe(MPI_ANY_SOURCE, THREAD_PREPARE_TAG, MPI_COMM_WORLD,
                             &isThere, &incoming_message));
        if (isThere)
        {
            int peer = incoming_message.MPI_SOURCE;
            int remoteRequestID = -1;
            force_mpi(MPI_Recv(&remoteRequestID, 1, MPI_INT,
                               peer, THREAD_PREPARE_TAG,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            std::cout << "Looking up: " << peer << " " << remoteRequestID << std::endl;
            remote_partner_to_local[std::make_tuple(peer,remoteRequestID)]->ready();
        }
    }
};

#endif