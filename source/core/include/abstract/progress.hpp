#ifndef ST_ABSTRACT_PROGRESS
#define ST_ABSTRACT_PROGRESS

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "safety/mpi.hpp"

namespace Progress
{
using MPIError       = int;
using CounterType    = int64_t;
using ProgressLambda = std::function<MPIError()>;

class Entry
{
public:
    Entry(ProgressLambda _action, CounterType* _mem)
        : action(std::move(_action)), mem_signal(_mem)
    {
    }

    virtual void enqueued_action() = 0;

    virtual void set_iteration(CounterType _iteration)
    {
        iteration = _iteration;
    }

protected:
    ProgressLambda action;
    CounterType*   mem_signal;
    CounterType    iteration;
};

class StartEntry : public Entry
{
public:
    StartEntry(ProgressLambda _action, CounterType* _mem)
        : Entry(std::move(_action), _mem)
    {
    }
    void enqueued_action() override
    {
        while ((*mem_signal) != iteration)
        {
            std::this_thread::yield();
        }
        force_mpi(action());
    }
};

class WaitEntry : public Entry
{
public:
    WaitEntry(ProgressLambda _action, CounterType* _mem) : Entry(std::move(_action), _mem)
    {
    }

    void enqueued_action() override
    {
        force_mpi(action());
        (*mem_signal) = iteration;
    }
};

class Engine
{
public:
    Engine() : running(false) {}

    ~Engine()
    {
        running = false;
        progress_thread.join();
    }

    void enqueued_start(std::shared_ptr<StartEntry> request, CounterType iteration)
    {
        request->set_iteration(iteration);

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