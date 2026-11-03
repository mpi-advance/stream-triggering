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
 * @brief Helper class designed to coalesce C++ "request abstractions" into a single array
 * for MPI functions.
 * @details Essentially a fancy container for a contiguous vector of MPI_Requests. This is
 * in contrast to what other objects may hold, such as a vector of shared pointers to MPI
 * Request objects, which are not contiguous at the MPI Request level.
 */
class RequestGroup
{
public:
    /**
     * @brief Create the contiguous container from an existing vector-based setup.
     * @details If any of the members of the existing vector are empty or null,
     * MPI_REQUEST_NULL is added to the vector to keep status object alignment. Note that
     * this should only happen for wait-based actions. The original list of members is
     * saved to Progress::RequestGroup::members to keep the shared pointers alive in case
     * the originals go away early.
     * @param [in] _members The vector that holds all of the MPI Requests to coalesce.
     * Currently comes in as a vector of Progress::RequestType objects.
     */
    explicit RequestGroup(std::vector<RequestType> _members)
        : members(std::move(_members))
    {
        handles.reserve(members.size());
        for (RequestType& member : members)
        {
            handles.push_back(member ? *member : MPI_REQUEST_NULL);
        }
    }

    /** @brief Copy constructor is deleted. */
    RequestGroup(const RequestGroup& other) = delete;
    /** @brief Copy assignment operator is deleted. */
    RequestGroup& operator=(const RequestGroup& other) = delete;

    /** @brief Get the number of MPI Request operations held by this object.
     * @return The number of underlying MPI Request operations.
     */
    int size()
    {
        return handles.size();
    }

    /** @brief Get the pointer of the underlying MPI_Request array.
     * @return The pointer to the underlying MPI_Request array.
     */
    MPI_Request* data()
    {
        return handles.data();
    }

private:
    /** @brief Vector of the original handles.*/
    std::vector<RequestType> members;
    /** @brief Coalesced underlying MPI Request handles. */
    std::vector<MPI_Request> handles;
};

/** @brief Abstract class that describes the interface for objects that can be given to a
 * progress engine.
 * @details The design of this class (and its interaction with Progress::Engine) is
 * modeled after how the CXI-based backend works, despite that backend not using anything
 * in the Progress namespace. Generally, each Progress::Entry has a "counter"
 * (Progress::Entry::mem_signal) that can be triggered or waited on. This class can then
 * be thought of as having a design loosely based on the libfabric deferred work queue
 * entry. However, the current implementation revolves around a set of underlying MPI
 * Request objects (via Entry::group) that each Entry is responsible for interacting with.
 */
class Entry
{
public:
    /** @brief Basic base constructor that just saves core member variables.
     * @param [in] _members The vector of Progress::RequestType objects to coalesce.
     * @param [in] _mem The address of the "counter"
     * @param [in] _iteration An iteration value whose meaning can vary based on
     * implementation.
     */
    Entry(std::vector<RequestType> _members, CounterType* _mem, CounterType _iteration)
        : group(std::move(_members)), mem_signal(_mem), iteration(_iteration)
    {
    }

    /** @brief Virtual, default destructor */
    virtual ~Entry() = default;

    /** @brief The function the Progress::Engine will call to invoke work to be done. */
    virtual void enqueued_action() = 0;

protected:
    /** @brief A set of underlying (contiguous) MPI Request actions to interact with. */
    RequestGroup group;
    /** @brief A "counter" address */
    CounterType* mem_signal;
    /** @brief The current iteration (full meaning is left up to derived classes) */
    CounterType iteration;
};

/** @brief An implementation of Progress::Entry that represents "starting" a triggered
 * communication operation
 * @details This derived class only deviates from Entry by defining
 * StartEntry::enqueued_action(). Progress::Entry::iteration is used as the threshold to
 * meet for the "triggered operation" to fire.
 */
class StartEntry : public Entry
{
public:
    /** @brief Basic derived constructor that passes values along to parent constructor.
     * @param [in] _members The vector of Progress::RequestType objects to coalesce.
     * @param [in] _mem The address of the "counter"
     * @param [in] _iteration The triggering threshold for this operation.
     */
    StartEntry(std::vector<RequestType> _members, CounterType* _mem,
               CounterType _iteration)
        : Entry(std::move(_members), _mem, _iteration)
    {
    }

    /** @brief Call MPI_Startall on the underlying MPI_Requests in Entry::group, when
     * triggered.
     * @details This function will sit in a while loop waiting for the expected value (or
     * a larger value) in Entry::mem_signal. Once there, then MPI_Startall is called.
     */
    void enqueued_action() override
    {
        while (std::atomic_ref<CounterType>(*mem_signal).load(std::memory_order_acquire) <
               iteration)
        {
            std::this_thread::yield();
        }
        force_mpi(MPI_Startall(group.size(), group.data()));
    }
};

/** @brief An implementation of Progress::Entry that represents "waiting" on a
 * communication operation on a queue.
 * @details This derived class only deviates from Entry by defining
 * WaitEntry::enqueued_action() and holding a pointer to an array of MPI_Status objects
 * for the corresponding MPI_Waitall. Anything calling this class and passing an
 * MPI_Status object should keep the array alive while this object is alive.
 *
 * Progress::Entry::iteration is used as the value to set in Progress::Entry::mem_signal
 * once the wait has completed.
 */
class WaitEntry : public Entry
{
public:
    /** @brief Basic derived constructor that passes values along to parent constructor.
     * @param [in] _members The vector of Progress::RequestType objects to coalesce.
     * @param [in] _mem The address of the "counter"
     * @param [in] _iteration The threshold value to write when the wait is done.
     * @param [in] _statuses Pointer to the array of statuses to eventually fill in.
     */
    WaitEntry(std::vector<RequestType> _members, CounterType* _mem,
              CounterType _iteration, MPI_Status* _statuses = MPI_STATUSES_IGNORE)
        : Entry(std::move(_members), _mem, _iteration), statuses(_statuses)
    {
    }

    /** @brief Call MPI_Waitall on the underlying MPI_Requests in Entry::group, then
     * trigger Progress::Entry::mem_signal.
     * @details Will write to the status array saved in WaitEntry::statuses. After
     * waiting, will put Progress::Entry::iteration into Progress::Entry::mem_signal.
     */
    void enqueued_action() override
    {
        force_mpi(MPI_Waitall(group.size(), group.data(), statuses));
        std::atomic_ref<CounterType>(*mem_signal)
            .store(iteration, std::memory_order_release);
    }

private:
    /** @brief Pointer to the array of statuses to fill in. */
    MPI_Status* statuses;
};

/** @brief A progress engine (thread) for MPI-based backends.
 * @details Does not spawn the thread until an operation is enqueued. Once the thread is
 * started, it runs until this object goes away.
 */
class Engine
{
public:
    /** @brief Basic constructor, does not start progress thread. */
    Engine() : running(false), item_counter(0) {}

    /** @brief Turns Progress::Engine::running to false, then waits on thread to
     * complete.
     * @details Blocking due to progress thread blocking on all outstanding operations
     * before completing.
     * @sa Progress::Engine::drain()
     **/
    ~Engine()
    {
        running = false;
        if (progress_thread.joinable())
        {
            progress_thread.join();
        }
    }

    /** @brief Enqueue an action to the progress engine.
     * @details The item must be of type Progress::Entry, since the thread will call
     * Progress::Entry::enqueued_action() on it (when it is that object's turn). Note the
     * similar name of the functions, but this function is present tense since it's
     * ongoing, while the object's function is past tense, since it was enqueued.
     *
     * Will start the progress thread if it is not running and will wait to acquire lock
     * before enqueuing item.
     * @param [in] item A unique pointer to the Progress::Entry to enqueue (and consume).
     */
    void enqueue_action(std::unique_ptr<Entry> item)
    {
        if (!running.exchange(true))
        {
            progress_thread = std::thread(&Engine::progress, this);
        }

        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        ongoing.push(std::move(item));
        item_counter++;
    }

    /** @brief Busy wait (while yielding thread) until no items are being worked on. */
    void wait_until_empty()
    {
        while (item_counter.load() != 0)
        {
            std::this_thread::yield();
        }
    }

    /** @brief Check if the progress engine has items to work on.
     * @return True if there are no enqueued (or ongoing) actions in the progress engine
     * queue, false otherwise.
     */
    bool is_empty()
    {
        return 0 == item_counter.load();
    }

private:
    /** @brief The main function for the progress thread.
     * @details While it is allowed to run, it attempts to do one cycle over all items
     * before yielding the thread. After finishing, it will do one more drain to give
     * everything one last chance to finish (note that this is still blocking).
     * The code structure does allow a cycle to work on things that were added during the
     * cycle; the lock is only over getting the next item off the queue, not during its
     * action.
     */
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

    /** @brief Iterate over all enqueued actions and attempt to do them.
     * @details This method works on the items, but doesn't do the heavy lifting to figure
     * out the "next item". See Progress::Engine::pop_next().
     *
     * Note that doing each action is "blocking", akin to a GPU queue.
     * The code has been set up to get around this with a "start all" and "wait all"
     * approach, but if designing a new Progress::Entry, be mindful of how this function
     * works.
     * Decreases Progress::Engine::item_counter after the invoked
     * Progress::Entry::enqueued_action() returns.
     */
    void drain()
    {
        while (std::unique_ptr<Entry> item = pop_next())
        {
            item->enqueued_action();
            item_counter--;
        }
    }

    /** @brief Get the next item for the progress thread to work on.
     * @details Does not change Progress::Engine::item_counter when it pops off
     * Progress::Engine::ongoing. Will acquire lock when attempting to pop.
     * @return A unique pointer to the next Progress::Entry to work on, or nullptr if
     * there are no more.
     */
    std::unique_ptr<Entry> pop_next()
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        if (ongoing.empty())
        {
            return nullptr;
        }
        std::unique_ptr<Entry> item = std::move(ongoing.front());
        ongoing.pop();
        return item;
    }

    /** @brief An atomic boolean that controls if the progress thread is to run or not. */
    std::atomic<bool> running;
    /** @brief The physical progress thread object. */
    std::thread progress_thread;

    /** @brief Atomic integer for counting the number of currently enqueued actions. */
    std::atomic<int> item_counter;
    /** @brief Mutex for controlling access to Progress::Engine::ongoing. */
    std::mutex queue_guard;
    /** @brief The "queue" for the progress engine. */
    std::queue<std::unique_ptr<Entry>> ongoing;
};

}  // namespace Progress

#endif
