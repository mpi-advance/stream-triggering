#ifndef ST_ABSTRACT_QUEUE_ENTRY
#define ST_ABSTRACT_QUEUE_ENTRY

#include <memory>

#include "progress.hpp"
#include "request.hpp"

using namespace Communication;

/**
 * @file entry.hpp
 * @brief Header file for the base QueueEntry class.
 */

/**
 * @brief A high-level object for describing something that can be enqueued to a Queue.
 * @details Common class for MPI-based backends (Thread, HIP, CUDA) for managing the
 * persistent communication operations that have come to the Queue. These backends often
 * make use of a progress engine, so this class acts as a baseline way to create
 * standardized objects for the thread to work on. This class is currently not explicitly
 * used by the CXI-based backend except for collective operations (CXI support for
 * collectives is considered experimental at this time).
 */
class QueueEntry
{
public:
    /**
     * @brief Constructor for creating a QueueEntry.
     * @details This method will create the internal MPI_Request used to carry out the
     * communication operation described by the passed in Communication::Request. The
     * internal MPI Request object is created with a custom shared pointer deleter that
     * takes care of freeing the MPI Request when there are no other references to it.
     *
     * The Progress::CounterType pointers default to nullptr if not provided.
     * @param [in] req The base Communication::Request this QueueEntry represents.
     * @param [in] start An optional address to use as the location of the "start"
     * trigger.
     * @param [in] wait An optional address to use as the location of the "wait" trigger.
     */
    explicit QueueEntry(std::shared_ptr<Request> req,
                        Progress::CounterType*   start = nullptr,
                        Progress::CounterType*   wait  = nullptr)
        : threshold(0),
          mpi_request(new MPI_Request(MPI_REQUEST_NULL),
                      [](MPI_Request* p) {
                          if (MPI_REQUEST_NULL != *p)
                          {
                              check_mpi(MPI_Request_free(p));
                          }
                          delete p;
                      }),
          original_request(req),
          start_location(start),
          wait_location(wait)
    {
        initialize_mpi_request(*req, mpi_request.get());
    }

    /** @brief Default destructor. */
    virtual ~QueueEntry() = default;

    /** @brief Copy Constructor Deleted: QueueEntry owns an MPI_Request. */
    QueueEntry(const QueueEntry& other) = delete;
    /** @brief Copy Assignment Operator Deleted: QueueEntry owns an MPI_Request. */
    QueueEntry& operator=(const QueueEntry& other) = delete;
    /** @brief Move Constructor Deleted (don't want to move around MPI_Request objects). */
    QueueEntry(QueueEntry&& other) = delete;
    /** @brief Move Assignment Operator Deleted (don't want to move around MPI_Request
     * objects). */
    QueueEntry& operator=(QueueEntry&& other) = delete;

    /**
     * @brief Create a start-action work item for the progress engine.
     * @details This method uses QueueEntry::mpi_request, QueueEntry::start_location, and
     * QueueEntry::threshold to produce a progress item for the progress thread.
     * @return A shared pointer to a Progress::StartEntry object to be queued up (to a
     * Progress::Engine).
     * @sa Progress::StartEntry
     */
    operator std::shared_ptr<Progress::StartEntry>()
    {
        return std::make_shared<Progress::StartEntry>(mpi_request, start_location,
                                                      threshold);
    }

    /**
     * @brief Create a wait-action work item for the progress engine.
     * @details This method uses QueueEntry::mpi_request, QueueEntry::wait_location, and
     * QueueEntry::threshold to produce a progress item for the progress thread.
     * @return A shared pointer to a Progress::WaitEntry object to be queued up (to a
     * Progress::Engine).
     * @sa Progress::WaitEntry
     */
    operator std::shared_ptr<Progress::WaitEntry>()
    {
        return std::make_shared<Progress::WaitEntry>(mpi_request, wait_location,
                                                     threshold);
    }

    /**
     * @brief Create a wait-all-action item for the progress engine.
     * @details This method is intended to allow for waiting on multiple items at once in
     * a single action. Currently this requires the list of MPI_Requests to be waited on
     * to be passed in. Invocation is a bit unwieldy at the moment, since the caller must
     * also include this QueueEntry::mpi_request object, and be able to work about the
     * fact that it will use this object's QueueEntry::wait_location and
     * QueueEntry::threshold. The resulting Progress::WaitallEntry behaves like the other
     * actions.
     * @param [in] other_reqs A vector of MPI_Request objects to be waited on in the
     * action.
     * @return A shared pointer to a Progress::WaitallEntry object to be queued up (to a
     * Progress::Engine).
     * @sa HIPQueue::enqueue_waitall()
     */
    std::shared_ptr<Progress::WaitallEntry> convert_to_waitall(
        std::vector<Progress::RequestType> other_reqs)
    {
        return std::make_shared<Progress::WaitallEntry>(other_reqs, wait_location,
                                                        threshold);
    }

    /**
     * @brief Increment the threshold variable of the QueueEntry.
     * @details This is ideally called each time the QueueEntry is "started". This
     * threshold is the "triggering" value needs to be, as well as what the "wait
     * location" completion value needs to be (in the general case).
     */
    virtual void increment()
    {
        threshold++;
    }

    /** @brief Get the threshold value of this QueueEntry.
     * @return The current threshold value.
     */
    virtual Progress::CounterType get_threshold()
    {
        return threshold;
    }

    /**
     * @brief Add a start operation to the given GPU stream.
     * @details Will throw an exception if this version is called.
     * @param [in] stream The GPU "stream" to put actions on. Meaning is determined by
     * derived class.
     */
    virtual void start_gpu(void* stream)
    {
        // Does nothing in base class.
        throw std::runtime_error("Function not supported: QueueEntry::start_gpu");
    }

    /**
     * @brief Add a wait operation to the given GPU stream.
     * @details Will throw an exception if this version is called.
     * @param [in] stream The GPU "stream" to put actions on. Meaning is determined by
     * derived class.
     */
    virtual void wait_gpu(void* stream)
    {
        // Does nothing in base class.
        throw std::runtime_error("Function not supported: QueueEntry::wait_gpu");
    }

    /** @brief Retrieve the address of the "counter" to trigger the start.
     * @return A pointer to the "counter" for triggering the start.
     * @sa Progress::CounterType
     */
    Progress::CounterType* get_start_location()
    {
        return start_location;
    }

    /** @brief Retrieve the address of the "counter" used in the wait.
     * @return A pointer to the "counter" for using with the wait.
     * @sa Progress::CounterType
     */
    Progress::CounterType* get_wait_location()
    {
        return wait_location;
    }

    /** @brief Retrieve a shared pointer to the underlying MPI_Request object.
     * @return A shared pointer to the underlying MPI_Request.
     */
    std::shared_ptr<MPI_Request> get_mpi_request()
    {
        return mpi_request;
    }

protected:
    /** @brief The triggering threshold for this communication operation. */
    Progress::CounterType threshold = 0;

    /**
     * @brief A shared pointer to the underlying MPI Request used for communication.
     * @details The MPI_Request is wrapped in a shared pointer so that the
     * Queue::progress_engine and the main thread can both read from it / manage the
     * object's lifetime safely. The main thread creates the shared pointer. Note that
     * this shared pointer is created with a custom deleter that will call
     * MPI_Request_free upon the last shared pointer going away.
     * @sa initialize_mpi_request
     */
    std::shared_ptr<MPI_Request> mpi_request = nullptr;

    /**
     * @brief A shared pointer to the original Communication::Request object.
     * @details Mostly kept around so a QueueEntry object can refer back to information
     * from the original request.
     */
    std::shared_ptr<Request> original_request = nullptr;

    /**
     * @brief The location the "stream" will write to trigger the start of some
     * communication.
     * @details The concept of a "start location" is modeled after the CXI backend, where
     * the GPU stream touches a memory location and a different object gets triggered by
     * this change to move data over the network. For this class, the other object is
     * likely the Queue::progress_engine. The exact usage of this variable depends on the
     * backend, and note that receive operations for the MPI-based backends are
     * (currently) triggered with this value (unlike the CXI backend).
     * @sa Progress::StartEntry
     * @sa ThreadQueue::enqueue_operation()
     * @sa HIPQueue::enqueue_operation()
     * @sa HIPQueueEntry::start_gpu()
     */
    Progress::CounterType* start_location;

    /**
     * @brief The location the "stream" will look at to wait on a communication.
     * @details Because the Queue::progress_engine is the one working on the completion of
     * the underlying MPI operation (and that the "stream" is assumed not to be able to
     * call MPI functions), this address is how the two communication on the completion of
     * the operation. The stream will wait until the value at this address hits a certain
     * value (likely QueueEntry::threshold). Currently,
     * Progress::WaitEntry::enqueued_action is the location of where this value is set for
     * all derived classes.
     * @sa Progress::WaitEntry()
     * @sa ThreadQueue::enqueue_waitall()
     * @sa HIPQueue::enqueue_waitall()
     * @sa HIPQueueEntry::wait_gpu()
     */
    Progress::CounterType* wait_location;

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
    void initialize_mpi_request(Request& req, MPI_Request* req_addr)
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