#ifndef ST_ABSTRACT_PROGRESS
#define ST_ABSTRACT_PROGRESS

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "misc/print.hpp"
#include "safety/mpi.hpp"

/**
 * @file progress.hpp
 * @brief Header file for the progress related classes and variables.
 */

/** @brief Namespace for all items related to progress (engines)*/
namespace Progress
{
/** @brief Helper alias type for MPI_Request ownership. */
using RequestType = std::shared_ptr<MPI_Request>;
/** @brief Helper alias type to make code more understandable. */
using CounterType = int64_t;

/**
 * @brief Abstract class that describes the interface for objects that can be given to a
 * progress engine.
 * @details The design of this class (and its interaction with Progress::Engine) is
 * modeled after how the CXI-based backend works, despite that backend not using anything
 * in the Progress namespace. Generally, each Progress::Entry has a "counter"
 * (Progress::Entry::mem_signal) that can be triggered or waited on. This class can then
 * be thought of a having a design loosely based on the libfabric deferred work queue
 * entry.
 */
class Entry
{
public:
    /**
     * @brief Basic base constructor that just saves core member variables.
     * @param [in] _mem The address of the "counter"
     * @param [in] _iteration An interation value whose meaning can vary based on
     * implementation.
     */
    Entry(CounterType* _mem, CounterType _iteration)
        : mem_signal(_mem), iteration(_iteration)
    {
    }

    /** @brief The function the Progress::Engine will call to invoke work to be done. */
    virtual void enqueued_action() = 0;

protected:
    CounterType* mem_signal;  //!< A "counter" address.
    CounterType  iteration;   //!< The current iteration (full meaning is left up to derived classes).
};

/**
 * @brief An implementation of Progress::Entry that represent triggered communication "starts"
 * @details
 */
class StartEntry : public Entry
{
public:
    /**
     * @brief
     * @param [in] _request
     * @param [in] _mem
     * @param [in] _iteration
     */
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
    /** @brief The underlying MPI Request that will be started by this Entry */
    RequestType request;
};

/**
 * @brief
 * @details
 */
class WaitEntry : public Entry
{
public:
    /**
     * @brief
     * @param [in] _request
     * @param [in] _mem
     * @param [in] _iteration
     */
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
    /** @brief The underlying MPI Request that will be waited on by this Entry */
    RequestType request;
};

/**
 * @brief
 * @details
 */
class WaitallEntry : public WaitEntry
{
public:
    /**
     * @brief
     * @param [in] _requests
     * @param [in] _mem
     * @param [in] _iteration
     */
    WaitallEntry(std::vector<RequestType> _requests, CounterType* _mem,
                 CounterType _iteration)
        : WaitEntry(_requests.at(0), _mem, _iteration), requests(_requests)
    {
        for (auto& x : requests)
        {
            true_requests.push_back(*x);
        }
    }

    void enqueued_action() override
    {
        force_mpi(
            MPI_Waitall(true_requests.size(), true_requests.data(), MPI_STATUSES_IGNORE));
        (*mem_signal) = iteration;
    }

private:
    std::vector<MPI_Request> true_requests;  //!<
    std::vector<RequestType> requests;       //!<
};

/**
 * @brief
 * @details
 */
class Engine
{
public:
    /** @brief */
    Engine() : running(false), item_counter(0) {}

    /** @brief */
    ~Engine()
    {
        running = false;
        if (progress_thread.joinable())
        {
            progress_thread.join();
        }
    }

    /**
     * @brief
     * @param [in] request
     */
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

    /**
     * @brief
     * @param [in] request
     */
    void enqueued_wait(std::shared_ptr<WaitEntry> request)
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        ongoing.push(request);
        item_counter++;
    }

    /** @brief */
    void wait_until_empty()
    {
        while (item_counter.load() != 0)
        {
            /* Do nothing */
        }
    }

private:
    /** @brief */
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

    bool        running;          //!<
    std::thread progress_thread;  //!<

    std::atomic<int>                   item_counter;  //!<
    std::mutex                         queue_guard;   //!<
    std::queue<std::shared_ptr<Entry>> ongoing;       //!<
};

}  // namespace Progress

#endif