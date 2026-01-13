#ifndef ST_ABSTRACT_QUEUE_ENTRY
#define ST_ABSTRACT_QUEUE_ENTRY

#include <memory>

#include "progress.hpp"
#include "request.hpp"

using namespace Communication;

// Common class for MPI-based backends (Thread, HIP, CUDA)
class QueueEntry
{
public:
    explicit QueueEntry(std::shared_ptr<Request> req)
        : threshold(0), original_request(req)
    {
        switch (req->operation)
        {
            case Communication::Operation::SEND:
                check_mpi(MPI_Send_init(req->buffer, req->count, req->datatype, req->peer,
                                        req->tag, req->comm, &mpi_request));
                break;
            case Communication::Operation::RSEND:
                check_mpi(MPI_Rsend_init(req->buffer, req->count, req->datatype,
                                         req->peer, req->tag, req->comm, &mpi_request));
                break;
            case Communication::Operation::RECV:
                check_mpi(MPI_Recv_init(req->buffer, req->count, req->datatype, req->peer,
                                        req->tag, req->comm, &mpi_request));
                break;
            case Communication::Operation::BARRIER:
                check_mpi(MPI_Barrier_init(req->comm, req->info, &mpi_request));
                break;
            case Communication::Operation::ALLREDUCE:
                // check_mpi(MPI_Allreduce_init(req->sendbuf, req->recvbuf, req->count,
                //                             req->datatype, req->op, req->comm,
                //                             req->info, &mpi_request));
                break;
            default:
                throw std::runtime_error("Invalid Request");
                break;
        }
    }

    virtual ~QueueEntry()
    {
        if (MPI_REQUEST_NULL != mpi_request && original_request)
        {
            check_mpi(MPI_Request_free(&mpi_request));
        }
    }

    // No copying
    QueueEntry(const QueueEntry& other)            = delete;
    QueueEntry& operator=(const QueueEntry& other) = delete;

    // Only moving
    QueueEntry(QueueEntry&& other) noexcept
        : threshold(other.threshold),
          mpi_request(other.mpi_request),
          original_request(other.original_request),
          start_lambda(other.start_lambda),
          start_location(other.start_location),
          wait_lambda(other.wait_lambda),
          wait_location(other.wait_location)
    {
        // clear other structs
        other.threshold   = 0;
        other.mpi_request = MPI_REQUEST_NULL;
        other.original_request.reset();
        other.start_lambda.reset();
        other.start_location = nullptr;
        other.wait_lambda.reset();
        other.wait_location = nullptr;

        // Reestablish lambdas
        initialize_lambdas();
    }
    QueueEntry& operator=(QueueEntry&& other) noexcept
    {
        if (this != &other)
        {
            threshold        = other.threshold;
            mpi_request      = other.mpi_request;
            original_request = other.original_request;
            start_lambda     = other.start_lambda;
            start_location   = other.start_location;
            wait_lambda      = other.wait_lambda;
            wait_location    = other.wait_location;
            // clear other
            other.threshold   = 0;
            other.mpi_request = MPI_REQUEST_NULL;
            other.original_request.reset();
            other.start_lambda.reset();
            other.start_location = nullptr;
            other.wait_lambda.reset();
            other.wait_location = nullptr;

            // Reestablish lambdas
            initialize_lambdas();
        }
        return *this;
    }

    operator std::shared_ptr<Progress::StartEntry>()
    {
        return start_lambda;
    }

    operator std::shared_ptr<Progress::WaitEntry>()
    {
        return wait_lambda;
    }

    virtual void initialize_lambdas()
    {
        start_lambda = std::make_shared<Progress::StartEntry>(
            [this]() { return MPI_Start(&mpi_request); }, start_location);
        wait_lambda = std::make_shared<Progress::WaitEntry>(
            [this]() { return MPI_Wait(&mpi_request, MPI_STATUS_IGNORE); },
            wait_location);
    }

    virtual Progress::CounterType increment()
    {
        return (threshold++);
    }

    virtual void start_gpu(void* stream)
    {
        // Does nothing in base class.
        throw std::runtime_error("Function not supported: QueueEntry::start_gpu");
    }

    virtual void wait_gpu(void* stream)
    {
        // Does nothing in base class.
        throw std::runtime_error("Function not supported: QueueEntry::wait_gpu");
    }

    Progress::CounterType* get_start_location()
    {
        return start_location;
    }

    Progress::CounterType* get_wait_location()
    {
        return wait_location;
    }

protected:
    Progress::CounterType threshold = 0;

    MPI_Request              mpi_request      = MPI_REQUEST_NULL;
    std::shared_ptr<Request> original_request = nullptr;

    std::shared_ptr<Progress::StartEntry> start_lambda;
    Progress::CounterType*                start_location;
    std::shared_ptr<Progress::WaitEntry>  wait_lambda;
    Progress::CounterType*                wait_location;
};

#endif