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

/** [REVIEW]
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
    /** [UPDATE]
     * @brief Virtual destructor.
     */
    virtual ~Queue()
    {
        /* Progress items point at the counters, so finish them before freeing */
        progress_engine.wait_until_empty();
        counter_resource->deallocate(start_signal, 2 * sizeof(Progress::CounterType),
                                     alignof(Progress::CounterType));
    }

    /** [UPDATE]
     * @brief Enqueue a start operation onto a Queue.
     * @param [in, out] req The Communication::Request to enqueue the start of.
     */
    virtual void enqueue_operation(std::shared_ptr<Request> req)
    {
        enqueue_startall({req});
    }

    /** [UPDATE]
     * @brief Enqueue a start for a set of operations onto a Queue.
     * @details While the provided base version is a simple, in-order loop over the vector
     * calling Queue::enqueue_operation(), the actual starting order of the communication
     * operations is not guaranteed.
     * @param [in, out] reqs The vector of Communication::Request objects to enqueue the
     * start of.
     */
    virtual void enqueue_startall(std::vector<std::shared_ptr<Request>> reqs) = 0;

    /**
     * @brief [TODO]
     * @param [in, out] req
     * @param [out] status
     */
    virtual void enqueue_wait(std::shared_ptr<Request> req, MPI_Status* status)
    {
        enqueue_waitall({req}, status);
    }

    /** [UPDATE]
     * @brief Enqueue a waitall operation to the Queue.
     * @param [in, out] reqs
     * @param [out] statuses
     */
    virtual void enqueue_waitall(std::vector<std::shared_ptr<Request>> reqs,
                                 MPI_Status*                           statuses) = 0;

    /**
     * @brief [TODO]
     * @return
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

    /** [REVIEW]
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
        auto found = request_cache.find(request->getID());
        if (request_cache.end() != found)
        {
            request->out("Removing from Queue's cache!");
            request_cache.erase(found);
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
     * @brief [TODO]
     * @param [in] _counter_resource
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

    /**
     * @brief [TODO]
     * @param [in] reqs
     * @return
     */
    Progress::CounterType build_start(const std::vector<std::shared_ptr<Request>>& reqs)
    {
        std::vector<Progress::RequestType> members;
        members.reserve(reqs.size());
        for (const std::shared_ptr<Request>& req : reqs)
        {
            members.push_back(find_or_create_request(*req));
        }

        progress_engine.enqueued_start(std::make_shared<Progress::StartEntry>(
            std::make_shared<Progress::RequestGroup>(std::move(members)), start_signal,
            ++start_target));
        return start_target;
    }

    /**
     * @brief [TODO]
     * @param [in] reqs
     * @param [out] statuses
     * @return
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

        progress_engine.enqueued_wait(std::make_shared<Progress::WaitEntry>(
            std::make_shared<Progress::RequestGroup>(std::move(members)), wait_signal,
            ++wait_target, (nullptr != statuses) ? statuses : MPI_STATUSES_IGNORE));
        return wait_target;
    }

    Progress::CounterType* start_signal = nullptr;  //!< [TODO]
    Progress::CounterType* wait_signal  = nullptr;  //!< [TODO]

    /** [UPDATE]
     * @brief A helper cache designed to keep track of all operations that have come
     * through.
     * @details More specifically, the map is intended to be a mapping between
     * Communication::Request::myID values and the corresponding QueueEntry object that
     * represents that persistent communication. Ideally, a derived class can make the
     * QueueEntry once, and then re-use it each time a communication operation is
     * re-started. Only used by the base class to remove operations in
     * Queue::request_free (and Queue::request_freeall).
     */
    std::map<size_t, Progress::RequestType> request_cache;

    /** [REVIEW]
     * @brief The optional progress engine for handling communication.
     * @details By default, this object will do nothing in any class that chooses not to
     * use it.
     */
    Progress::Engine progress_engine;

private:
    std::pmr::memory_resource* counter_resource;  //!< [TODO]
    Progress::CounterType      start_target = 0;  //!< [TODO]
    Progress::CounterType      wait_target  = 0;  //!< [TODO]

    /**
     * @brief [TODO]
     * @param [in] req
     * @return
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

    /**
     * @brief Internal method for creating the true underlying MPI operation.
     * @details Since this library sits on top of an MPI implementation, and because we
     * override the MPI_Send/Recv-based bindings, we still need some way to carry out the
     * MPI operation (for MPI-based Queues). This method uses a switch statement based off
     * the Communication::Operation inside the Communication::Request object.
     * The full list of supported operations matches what is documented in
     * Communication::Operation; wil throw an exception otherwise. Because this library
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