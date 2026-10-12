#ifndef ST_ABSTRACT_PROGRESS
#define ST_ABSTRACT_PROGRESS

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

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

    /** @brief [TODO] */
    virtual ~Entry() = default;

    /** @brief The function the Progress::Engine will call to invoke work to be done. */
    virtual void enqueued_action() = 0;

protected:
    CounterType* mem_signal;  //!< A "counter" address.
    CounterType iteration;  //!< The current iteration (full meaning is left up to derived
                            //!< classes).
};

/**
 * @brief An implementation of Progress::Entry that represent triggered communication
 * "starts"
 * @details
 */
class StartEntry : public Entry
{
public:
    /**
     * @brief [TODO]
     * @param [in] _request
     * @param [in] _mem
     * @param [in] _iteration
     */
    StartEntry(RequestType _request, CounterType* _mem, CounterType _iteration)
        : Entry(_mem, _iteration), request(std::move(_request))
    {
    }
    void enqueued_action() override
    {
        while (std::atomic_ref<CounterType>(*mem_signal).load(std::memory_order_acquire) <
               iteration)
        {
            std::this_thread::yield();
        }
        force_mpi(MPI_Start(request.get()));
    }

private:
    /** @brief The underlying MPI Request that will be started by this Entry */
    RequestType request;
};

/**
 * @brief [TODO]
 * @details
 */
class RequestGroup
{
public:
    /**
     * @brief [TODO]
     * @param [in] _members
     */
    explicit RequestGroup(std::vector<RequestType> _members)
        : members(std::move(_members))
    {
        handles.reserve(members.size());
        for (RequestType& member : members)
        {
            handles.push_back(*member);
        }
    }

    /**
     * @brief [TODO]
     * @param [in] _member
     */
    explicit RequestGroup(RequestType _member)
        : RequestGroup(std::vector<RequestType>{std::move(_member)})
    {
    }

    /** @brief [TODO] */
    RequestGroup(const RequestGroup& other) = delete;
    /** @brief [TODO] */
    RequestGroup& operator=(const RequestGroup& other) = delete;

    /** @brief [TODO]
     * @return
     */
    int size()
    {
        return handles.size();
    }

    /** @brief [TODO]
     * @return
     */
    MPI_Request* data()
    {
        return handles.data();
    }

private:
    std::vector<RequestType> members;  //!< [TODO]
    std::vector<MPI_Request> handles;  //!< [TODO]
};

/**
 * @brief [TODO]
 * @details
 */
class WaitEntry : public Entry
{
public:
    /**
     * @brief [TODO]
     * @param [in] _group
     * @param [in] _mem
     * @param [in] _iteration
     * @param [out] _statuses
     */
    WaitEntry(std::shared_ptr<RequestGroup> _group, CounterType* _mem,
              CounterType _iteration, MPI_Status* _statuses = MPI_STATUSES_IGNORE)
        : Entry(_mem, _iteration), group(std::move(_group)), statuses(_statuses)
    {
    }

    void enqueued_action() override
    {
        force_mpi(MPI_Waitall(group->size(), group->data(), statuses));
        std::atomic_ref<CounterType>(*mem_signal)
            .store(iteration, std::memory_order_release);
    }

private:
    std::shared_ptr<RequestGroup> group;     //!< [TODO]
    MPI_Status*                   statuses;  //!< [TODO]
};

/**
 * @brief [TODO]
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
     * @brief [TODO]
     * @param [in] request
     */
    void enqueued_start(std::shared_ptr<StartEntry> request)
    {
        push(std::move(request));
    }

    /**
     * @brief [TODO]
     * @param [in] request
     */
    void enqueued_wait(std::shared_ptr<WaitEntry> request)
    {
        push(std::move(request));
    }

    /** @brief [TODO] */
    void wait_until_empty()
    {
        while (item_counter.load() != 0)
        {
            std::this_thread::yield();
        }
    }

    /** @brief [TODO]
     * @return
     */
    bool is_empty()
    {
        return 0 == item_counter.load();
    }

private:
    /**
     * @brief [TODO]
     * @param [in] item
     */
    void push(std::shared_ptr<Entry> item)
    {
        if (!running.exchange(true))
        {
            progress_thread = std::thread(&Engine::progress, this);
        }

        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        ongoing.push(std::move(item));
        item_counter++;
    }

    /** @brief */
    void progress()
    {
        while (running)
        {
            drain();
            std::this_thread::yield();
        }
        /* Finish anything still queued when the engine was shut down */
        drain();
    }

    /** @brief [TODO] */
    void drain()
    {
        while (std::shared_ptr<Entry> item = pop_next())
        {
            item->enqueued_action();
            item_counter--;
        }
    }

    /** @brief [TODO]
     * @return
     */
    std::shared_ptr<Entry> pop_next()
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        if (ongoing.empty())
        {
            return nullptr;
        }
        std::shared_ptr<Entry> item = std::move(ongoing.front());
        ongoing.pop();
        return item;
    }

    std::atomic<bool> running;          //!< [TODO]
    std::thread       progress_thread;  //!<

    std::atomic<int>                   item_counter;  //!<
    std::mutex                         queue_guard;   //!<
    std::queue<std::shared_ptr<Entry>> ongoing;       //!<
};

}  // namespace Progress

#endif
