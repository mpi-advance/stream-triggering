#ifndef ST_ABSTRACT_QUEUE
#define ST_ABSTRACT_QUEUE

#include <stdint.h>

#include <map>
#include <memory>
#include <vector>

#include "entry.hpp"
#include "match.hpp"
#include "progress.hpp"
#include "request.hpp"

using namespace Communication;

/**
 * @file queue.hpp
 * @brief Header file for the abstract Queue class.
 */

/**
 * @brief A mostly abstract base class for all Queue classes.
 * @details This abstract class defines the methods that the bindings will use to carry
 * out the actions of the stream-triggering interface (enqueueing starts and waits,
 * matching requests, etc.).
 *
 * Some of the methods in this class are already filled out as they are either simple
 * loops around other methods, or have code currently common to all Queue implementations.
 *
 * All queues carry a dormant Progress::Engine, but the abstract class defines no way to
 * actually start it up. Likewise, helper containers are provided for derived classes to
 * use for managing ongoing/previous communication actions that have been given to the
 * Queue.
 */
class Queue
{
public:
    /** @brief Virtual destructor. */
    virtual ~Queue() = default;

    /**
     * @brief Enqueue a start operation onto a Queue.
     * @param [in, out] req The Communication::Request to enqueue the start of.
     */
    virtual void enqueue_operation(std::shared_ptr<Request> req) = 0;

    /**
     * @brief Enqueue a start for a set of operations onto a Queue.
     * @details While the provided base version is a simple, in-order loop over the vector
     * calling Queue::enqueue_operation(), the actual starting order of the communication
     * operations is not guaranteed.
     * @param [in, out] reqs The vector of Communication::Request objects to enqueue the
     * start of.
     */
    virtual void enqueue_startall(std::vector<std::shared_ptr<Request>> reqs)
    {
        for (auto& req : reqs)
        {
            enqueue_operation(req);
        }
    }

    /** @brief Enqueue a waitall operation to the Queue. */
    virtual void enqueue_waitall() = 0;

    /** @brief Block the host (caller) until the Queue is empty.
     * @details "Empty" in this case can vary depending on what the Queue's backend is.
     * For example, Queues representing GPU streams may perform a synchronize with the
     * underlying stream. Others may block until all MPI communications are done.
     */
    virtual void host_wait()
    {
        progress_engine.wait_until_empty();
    }

    /**
     * @brief Begin the match process for a set of Communication::Request objects.
     * @details This implementation calls Match::Blank::match for each request (that is a
     * point-to-point operation), in the order they were passed in. Derived methods are
     * free to choose any order, and may use different matching methods for each request.
     * Note that when this call returns, a communication should not considered fully
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

    /**
     * @brief Finalize the match between communication operations.
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
        std::vector<MPI_Status>  status_train;
        for (auto& req : requests)
        {
            if (Operation::BARRIER > req->operation)
            {
                req->join_waitall_match(request_train, status_train);
                Print::out("Train size now:", request_train.size());
            }
        }

        force_mpi(
            MPI_Waitall(request_train.size(), request_train.data(), status_train.data()));

        Print::out("Done with all matching.");
        for (auto& req : requests)
        {
            req->set_match();
        }
    }

    /**
     * @brief Free the request from any Queue resources.
     * @details For the base class, this involves removing the QueueEntry mapped to
     * Request::myID from Queue::request_cache so that any resources associated with the
     * QueueEntry can be freed up. This is needed since communication requests are
     * persistent in the stream-triggering library, so any un-freed request will hold its
     * resources until the program is terminated.
     * @param [in, out] request The Communication::Request to be removed.
     */
    virtual void request_free(std::shared_ptr<Request> request)
    {
        if (request_cache.contains(request->getID()))
        {
            request->out("Removing from Queue's cache!");
            request_cache.erase(request->getID());
        }
    }

    /**
     * @brief Free a set of Communication::Request objects.
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
    /**
     * @brief The optional progress engine for handling communication.
     * @details By default, this object will do nothing in any class that chooses not to
     * use it.
     */
    Progress::Engine progress_engine;

    /**
     * @brief A helper vector for keeping track of all ongoing operations.
     * @details Derived classes can use this vector to keep track of all the ongoing
     * communication (or communication adjacent) actions that have been "enqueued", though
     * this base class currently does nothing with it.
     */
    std::vector<std::reference_wrapper<QueueEntry>> entries;

    /**
     * @brief A helper cache designed to keep track of all operations that have come
     * through.
     * @details More specifically, the map is intended to be a mapping between
     * Communication::Request::myID values and the corresponding QueueEntry object that
     * represents that persistent communication. Ideally, a derived class can make the
     * QueueEntry once, and then re-use it each time a communication operation is
     * re-started. Only used by the base class to remove operations in
     * Queue::request_free (and Queue::request_freeall).
     */
    std::map<size_t, std::unique_ptr<QueueEntry>> request_cache;
};

#endif