#ifndef ST_ABSTRACT_QUEUE
#define ST_ABSTRACT_QUEUE

#include <stdint.h>

#include <map>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <vector>

#include "match.hpp"
#include "progress.hpp"
#include "request.hpp"

using namespace Communication;

/**
 * @file queue.hpp
 * @brief Header file for the abstract Queue class.
 */

/** @brief A mostly abstract base class for all Queue classes.
 * @details This abstract class defines the methods that the bindings will use to carry
 * out the actions of the stream-triggering interface (enqueueing starts and waits,
 * matching requests, etc.).
 *
 * Some of the methods in this class are already filled out as they are either simple
 * loops around other methods, or have code currently common to all Queue implementations.
 *
 * All queues carry a dormant Progress::Engine that will be started when
 * Queue::build_start or Queue::build_wait is called. Note that the base class does not
 * call either of these on its own; it is implied that the derived implementations of
 * Queue::enqueue_startall and Queue::enqueue_waitall will be the ones to do so (but do
 * not have to).
 */
class Queue
{
public:
    /** @brief Virtual destructor that waits for progress engine to finish before freeing
     * counter resources.
     */
    virtual ~Queue()
    {
        /* Progress items point at the counters, so finish them before freeing */
        progress_engine.wait_until_empty();
        counter_resource->deallocate(start_signal, 2 * sizeof(Progress::CounterType),
                                     alignof(Progress::CounterType));
    }

    /** @brief Enqueue a start operation onto a Queue.
     * @details Turns this single operation into a call to Queue::enqueue_startall().
     * @param [in, out] req The Communication::Request to enqueue the start of.
     */
    virtual void enqueue_operation(std::shared_ptr<Request> req)
    {
        enqueue_startall({req});
    }

    /** @brief Enqueue the start for a set of operations onto a Queue.
     * @details Default implementation not provided. Derived classes should figure out how
     * to do the startall, but thread-based implementations will likely use the provided
     * progress thread (Queue::progress_engine) and the Progress::Entry classes.
     * @param [in, out] reqs The vector of Communication::Request objects to enqueue the
     * start of.
     * @sa Queue::build_start()
     */
    virtual void enqueue_startall(std::vector<std::shared_ptr<Request>> reqs) = 0;

    /** @brief Enqueue a wait operation onto a Queue.
     * @details Turns this single operation into a call to Queue::enqueue_waitall().
     * The filling of status objects is not done by all backends (e.g. CXI).
     * @param [in, out] req The Communication::Request to enqueue the wait for.
     * @param [in] status A pointer to the MPI_Status to be filled in upon the completion
     * of the wait. Pointer should be valid until the wait is done.
     */
    virtual void enqueue_wait(std::shared_ptr<Request> req, MPI_Status* status)
    {
        enqueue_waitall({req}, status);
    }

    /** @brief Enqueue the wait for a set of operations onto a Queue.
     * @details Default implementation not provided. Derived classes should figure out how
     * to do the waitall, but thread-based implementations will likely use the provided
     * progress thread (Queue::progress_engine) and the Progress::Entry classes.
     * The filling of status objects is not done by all backends (e.g. CXI).
     * @param [in, out] reqs The vector of Communication::Request objects to enqueue the
     * waitall for.
     * @param [out] statuses A pointer to an array of MPI_Status objects to be filled in
     * upon the completion of the waitall. Pointer should be valid until the waitall is
     * done.
     * @sa Queue::build_wait()
     */
    virtual void enqueue_waitall(std::vector<std::shared_ptr<Request>> reqs,
                                 MPI_Status*                           statuses) = 0;

    /** @brief Test if the ::Queue is empty.
     * @details Default implementation just returns what Progress::Engine::is_empty()
     * returns, but derived classes should be sure to check on any associated external
     * "queue" objects.
     * @return False if the ::Queue is working on anything (stream-triggered communication
     * or otherwise), true otherwise.
     */
    virtual bool queue_test()
    {
        return progress_engine.is_empty();
    }

    /** @brief Block the host (caller) until the Queue is empty.
     * @details "Empty" in this case can vary depending on what the Queue's backend is.
     * For example, Queues representing GPU streams may perform a synchronize with the
     * underlying stream. Others may block until all MPI communications are done.
     */
    virtual void host_wait()
    {
        progress_engine.wait_until_empty();
    }

    /** @brief Begin the match process for a set of Communication::Request objects.
     * @details This implementation calls Match::Blank::match for each request (that is a
     * point-to-point operation), in the order they were passed in. Derived methods are
     * free to choose any order, and may use different matching methods for each request.
     * Note that when this call returns, a communication should not be considered fully
     * matched until it a Queue sees it in Queue::finalize_match(). This is to help cases
     * where matching needs to exchange preliminary data in order to do a proper match.
     * @param [in, out] requests A vector of shared pointers of the Communication::Request
     * that should be matched.
     * @sa Queue::finalize_match()
     */
    virtual void initiate_match(std::vector<std::shared_ptr<Request>> requests)
    {
        for (auto& req : requests)
        {
            if (Operation::BARRIER > req->operation)
            {
                // Normal matching
                Match::Blank::match(*req);
            }
        }
    }

    /** @brief Finalize the match between communication operations.
     * @details When returning from this function, all of the input communication
     * operations should have been matched with a corresponding communication operation on
     * the remote peer. At a minimum, Communication::Request::set_match() should have been
     * called on all communication operations passed to this function.
     *
     * The default implementation assumes that derived classes will use regular MPI calls
     * to carry out the matching of operations. As such, this method builds a vector
     * (called a "train") of any outstanding MPI_Requests and then performs a giant
     * MPI_Waitall on them all before finally calling Communication::Request::set_match()
     * on each request. This method is implemented in this way since having each request
     * call MPI_Waitall on their own requests would likely create deadlock stemming from
     * each process having a different order of MPI operations. (e.g., peers could block
     * on their own individual waits in different orders)
     *
     * @param [in, out] requests A vector of shared pointers of the Communication::Request
     * that should have their match process finalized.
     * @sa Queue::initiate_match()
     * @sa Communication::Request::join_waitall_match()
     */
    virtual void finalize_match(std::vector<std::shared_ptr<Request>> requests)
    {
        std::vector<MPI_Request> request_train;
        for (auto& req : requests)
        {
            if (Operation::BARRIER > req->operation)
            {
                req->join_waitall_match(request_train);
                Print::out("Train size now:", request_train.size());
            }
        }

        force_mpi(
            MPI_Waitall(request_train.size(), request_train.data(), MPI_STATUSES_IGNORE));

        Print::out("Done with all matching.");
        for (auto& req : requests)
        {
            req->set_match();
        }
    }

    /** @brief Free the request from any Queue resources.
     * @details For the base class, this involves removing the MPI_Request mapped to
     * Request::myID from Queue::request_cache so that any resources associated with the
     * MPI_Request can be freed up. Currently, the associated resources are managed
     * through a custom deleter in Progress::RequestType that will call MPI_Request_free
     * when the last reference goes away. This is needed since communication requests are
     * persistent in the stream-triggering library, so any un-freed request will hold its
     * resources until the program is terminated.
     * Because the values in the map are shared pointers, freeing them here does not free
     * them from the progress engine (if it happens to be working on them).
     * @param [in, out] request The Communication::Request to be removed.
     */
    virtual void request_free(std::shared_ptr<Request> request)
    {
        auto found = request_cache.find(request->getID());
        if (request_cache.end() != found)
        {
            request->out("Removing from Queue's cache!");
            request_cache.erase(found);
        }
    }

    /** @brief Free a set of Communication::Request objects.
     * @details Default implementation is just a loop over Queue::request_free().
     * @param [in, out] requests The set of Communication::Request objects to free.
     */
    virtual void request_freeall(std::vector<std::shared_ptr<Request>> requests)
    {
        for (auto& req : requests)
        {
            request_free(req);
        }
    }

protected:
    /** @brief Protected constructor for the ::Queue class
     * @details This constructor is protected since one should not be making this class
     * directly. It also allocates the two "counter signals" held by the ::Queue,
     * and sets them to zero, and they share the same allocation.
     * @param [in] _counter_resource The counter resource to allocate the counters from
     * (saved to Queue::counter_resource). Defaults to a CPU-based allocation method.
     */
    explicit Queue(
        std::pmr::memory_resource* _counter_resource = std::pmr::new_delete_resource())
        : counter_resource(_counter_resource)
    {
        start_signal  = static_cast<Progress::CounterType*>(counter_resource->allocate(
            2 * sizeof(Progress::CounterType), alignof(Progress::CounterType)));
        wait_signal   = start_signal + 1;
        *start_signal = 0;
        *wait_signal  = 0;
    }

    /** @brief Protected helper method for coalescing a set of Communication::Request
     * objects into a single Progress::StartEntry for the progress engine.
     * @details While this method does not return the created Progress::StartEntry object
     * (it's given to Queue::progress_engine), it is derived from the passed in
     * Communication::Request objects. The provided operations are first converted to
     * Progress::RequestType before being handed over to the Progress::StartEntry. If any
     * Communication::Request object is not in Queue::request_cache, a
     * Progress::RequestType (with a custom deleter) is made for it. Null shared pointers
     * should not be passed.
     * @param [in] reqs The (non-null) Communication::Request operations to have their
     * start enqueued.
     * @return The value to use to trigger the start of the enqueued start(all) operation.
     */
    Progress::CounterType build_start(const std::vector<std::shared_ptr<Request>>& reqs)
    {
        std::vector<Progress::RequestType> members;
        members.reserve(reqs.size());
        for (const std::shared_ptr<Request>& req : reqs)
        {
            members.push_back(find_or_create_request(*req));
        }

        progress_engine.enqueue_action(std::make_unique<Progress::StartEntry>(
            std::move(members), start_signal, ++start_target));
        return start_target;
    }

    /** @brief Protected helper method for coalescing a set of Communication::Request
     * objects into a single Progress::WaitEntry for the progress engine.
     * @details While this method does not return the created Progress::WaitEntry object
     * (it's given to Queue::progress_engine), it is derived from the passed in
     * Communication::Request objects. The provided operations are first converted to
     * Progress::RequestType before being handed over to the Progress::WaitEntry. If any
     * Communication::Request object is not in Queue::request_cache, or if any provided
     * shared pointers are null, a null pointer (later, MPI_REQUEST_NULL) is also passed
     * along (so that the array of status and requests are the same length).
     * @param [in] reqs The Communication::Request operations to have their wait enqueued.
     * @param [in] statuses A pointer to an array of MPI_Status objects to be filled in
     * upon the completion of the waitall. Pointer should be valid until the waitall is
     * done. Null pointer is mapped MPI_STATUS_IGNORE.
     * @return The value to check for to know that the enqueued wait(all) operation has
     * finished.
     */
    Progress::CounterType build_wait(const std::vector<std::shared_ptr<Request>>& reqs,
                                     MPI_Status* statuses)
    {
        std::vector<Progress::RequestType> members;
        members.reserve(reqs.size());
        for (const std::shared_ptr<Request>& req : reqs)
        {
            auto found = request_cache.find(req->getID());
            members.push_back((request_cache.end() != found) ? found->second : nullptr);
        }

        progress_engine.enqueue_action(std::make_unique<Progress::WaitEntry>(
            std::move(members), wait_signal, ++wait_target,
            (nullptr != statuses) ? statuses : MPI_STATUSES_IGNORE));
        return wait_target;
    }

    /** @brief The memory location to write to trigger the start of some action. */
    Progress::CounterType* start_signal = nullptr;
    /** @brief The memory location to watch for the completion of a wait action. */
    Progress::CounterType* wait_signal = nullptr;

    /** @brief A helper cache designed to keep track of all operations that have been
     * started at least once.
     * @details More specifically, the map is intended to be a mapping between
     * Communication::Request::myID values and the corresponding Progress::RequestType
     * object that represents that underlying persistent MPI communication. Ideally, this
     * object is made once and then is re-used each time a communication operation is
     * re-started. When the ::Queue object goes away, the map will be cleared and if the
     * stored shared pointers are the last references, the underlying MPI_Request will be
     * freed.
     * Added to in Queue::build_start(), removed from in Queue::request_free().
     */
    std::map<size_t, Progress::RequestType> request_cache;

    /** @brief The optional progress engine for handling communication.
     * @details By default, this object will do nothing in any class that chooses not to
     * use it.
     */
    Progress::Engine progress_engine;

private:
    /** @brief Pointer to the pool to allocate memory for the "counters" from.
     * @details Used by the constructor to safely create the memory it needs for
     * Queue::start_signal and Queue::wait_signal, and then by the destructor to free
     * them. This object does not align well to the "counters" that the CXIQueue will
     * create, and is purely for the MPI-based queue implementations.
     */
    std::pmr::memory_resource* counter_resource;
    /** @brief The threshold for most recent start operation to be triggered at. */
    Progress::CounterType start_target = 0;
    /** @brief The threshold for most recent previous wait operation to wait on. */
    Progress::CounterType wait_target = 0;

    /** @brief Determine if the provided Communication::Request has been used before, and
     * convert it to the appropriate Progress::RequestType object.
     * @details This method is created to convert from Communication::Request to the
     * corresponding MPI_Request (wrapped in Progress::RequestType). If the
     * Communication::Request has been seen before (Communication::Request::getID() is in
     * Queue::request_cache), the existing Progress::RequestType is pulled out (it's a
     * shared pointer, so really a copy of it) and returned.
     *
     * If the Communication::Request is NEW, then Queue::initialize_mpi_request() is called
     * to get the underlying MPI_Request, which is then wrapped in a Progress::RequestType
     * (custom deleter) before being inserted into the map and finally returned.
     *
     * @param [in] req The Communication::Request object to check
     * @return The corresponding Progress::RequestType that holds the underlying MPI
     * operation.
     */
    Progress::RequestType find_or_create_request(Request& req)
    {
        auto found = request_cache.find(req.getID());
        if (request_cache.end() == found)
        {
            Progress::RequestType mpi_request(new MPI_Request(MPI_REQUEST_NULL),
                                              [](MPI_Request* p) {
                                                  if (MPI_REQUEST_NULL != *p)
                                                  {
                                                      check_mpi(MPI_Request_free(p));
                                                  }
                                                  delete p;
                                              });
            initialize_mpi_request(req, mpi_request.get());
            found = request_cache.emplace(req.getID(), std::move(mpi_request)).first;
        }
        return found->second;
    }

    /** @brief Internal method for creating the true underlying MPI operation.
     * @details Since this library sits on top of an MPI implementation, and because we
     * override the MPI_Send/Recv-based bindings, we still need some way to carry out the
     * MPI operation (for MPI-based Queues). This method uses a switch statement based off
     * the Communication::Operation inside the Communication::Request object.
     * The full list of supported operations matches what is documented in
     * Communication::Operation; will throw an exception otherwise. Because this library
     * aims to be persistent, persistent MPI operation calls are used.
     *
     * @param [in] req The Communication::Request object that describes the
     * communication operation to be performed.
     * @param [out] req_addr The location of where the MPI_Request will go.
     */
    static void initialize_mpi_request(Request& req, MPI_Request* req_addr)
    {
        switch (req.operation)
        {
            case Communication::Operation::SEND:
                check_mpi(MPI_Send_init(req.send_buffer, req.count, req.datatype,
                                        req.peer, req.tag, req.comm, req_addr));
                break;
            case Communication::Operation::RSEND:
                check_mpi(MPI_Rsend_init(req.send_buffer, req.count, req.datatype,
                                         req.peer, req.tag, req.comm, req_addr));
                break;
            case Communication::Operation::RECV:
                check_mpi(MPI_Recv_init(req.recv_buffer, req.count, req.datatype,
                                        req.peer, req.tag, req.comm, req_addr));
                break;
            case Communication::Operation::BARRIER:
                check_mpi(MPI_Barrier_init(req.comm, req.info, req_addr));
                break;
            case Communication::Operation::ALLREDUCE:
                check_mpi(MPI_Allreduce_init(req.send_buffer, req.recv_buffer, req.count,
                                             req.datatype, req.op, req.comm, req.info,
                                             req_addr));
                break;
            default:
                throw std::runtime_error("Invalid Request");
                break;
        }
    }
};

#endif