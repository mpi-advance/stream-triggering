#ifndef ST_ABSTRACT_PROGRESS
#define ST_ABSTRACT_PROGRESS

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "misc/print.hpp"
#include "safety/mpi.hpp"

namespace Progress
{
using RequestType = std::shared_ptr<MPI_Request>;
using MPIError    = int;
using CounterType = int64_t;

class Entry
{
public:
    Entry(CounterType* _mem, CounterType _iteration)
        : mem_signal(_mem), iteration(_iteration)
    {
    }

    virtual void enqueued_action() = 0;

protected:
    CounterType* mem_signal;
    CounterType  iteration;
};

class StartEntry : public Entry
{
public:
    StartEntry(RequestType _request, CounterType* _mem, CounterType _iteration)
        : Entry(_mem, _iteration), request(_request)
    {
    }
    void enqueued_action() override
    {
        while ((*mem_signal) != iteration)
        {
            std::this_thread::yield();
            if ((*mem_signal) > iteration)
            {
                break;
            }
        }
        force_mpi(MPI_Start(request.get()));
    }

private:
    RequestType request;
};

class WaitEntry : public Entry
{
public:
    WaitEntry(RequestType _request, CounterType* _mem, CounterType _iteration)
        : Entry(_mem, _iteration), request(_request)
    {
    }

    void enqueued_action() override
    {
        force_mpi(MPI_Wait(request.get(), MPI_STATUS_IGNORE));
        (*mem_signal) = iteration;
    }

    private:
    RequestType request;
};

class Engine
{
public:
    Engine() : running(false), item_counter(0) {}

    ~Engine()
    {
        running = false;
        progress_thread.join();
    }

    void enqueued_start(std::shared_ptr<StartEntry> request)
    {
        if (!running)
        {
            running         = true;
            progress_thread = std::thread(&Engine::progress, this);
        }

        /* Add to progress thread */
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        ongoing.push(request);
        item_counter++;
    }

    void enqueued_wait(std::shared_ptr<WaitEntry> request)
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        ongoing.push(request);
        item_counter++;
    }

    void wait_until_empty()
    {
        while (item_counter.load() != 0)
        {
            /* Do nothing */
        }
    }

private:
    void progress()
    {
        while (running)
        {
            while (ongoing.size() > 0)
            {
                std::shared_ptr<Entry> item = std::move(ongoing.front());
                {  // Scope of the lock
                    std::scoped_lock<std::mutex> incoming_lock(queue_guard);
                    ongoing.pop();
                }

                item->enqueued_action();
                Print::always("Done.");
                item_counter--;
            }

            std::this_thread::yield();
        }
    }

    bool        running;
    std::thread progress_thread;

    std::atomic<int>                   item_counter;
    std::mutex                         queue_guard;
    std::queue<std::shared_ptr<Entry>> ongoing;
};

}  // namespace Progress

#endif