#ifndef ST_ABSTRACT_QUEUE_ENTRY
#define ST_ABSTRACT_QUEUE_ENTRY

#include "request.hpp"

using namespace Communication;

// Common class for MPI-based backends (Thread, HIP, CUDA)
class QueueEntry
{
public:
    explicit QueueEntry(std::shared_ptr<Request> req) : original_request(req)
    {
        switch (req->operation)
        {
            case Communication::Operation::SEND:
                check_mpi(MPI_Send_init(req->buffer, req->count, req->datatype,
                                        req->peer, req->tag, req->comm,
                                        &mpi_request));
                break;
            case Communication::Operation::RECV:
                check_mpi(MPI_Recv_init(req->buffer, req->count, req->datatype,
                                        req->peer, req->tag, req->comm,
                                        &mpi_request));
                break;
            case Communication::Operation::BARRIER:
                break;
            default:
                throw std::runtime_error("Invalid Request");
                break;
        }
    }

    virtual ~QueueEntry()
    {
        if (MPI_REQUEST_NULL != mpi_request && original_request &&
            original_request->operation != Communication::Operation::BARRIER)
        {
            check_mpi(MPI_Request_free(&mpi_request));
        }
    }

    // No copying
    QueueEntry(const QueueEntry& other)            = delete;
    QueueEntry& operator=(const QueueEntry& other) = delete;

    // Only Moving
    QueueEntry(QueueEntry&& other) noexcept
        : mpi_request(other.mpi_request),
          original_request(other.original_request)
    {
        // clear other structs
        other.mpi_request = MPI_REQUEST_NULL;
        other.original_request.reset();
    }
    QueueEntry& operator=(QueueEntry&& other) noexcept
    {
        if (this != &other)
        {
            mpi_request      = other.mpi_request;
            original_request = other.original_request;
            // clear other
            other.mpi_request = MPI_REQUEST_NULL;
            other.original_request.reset();
        }
        return *this;
    }

    virtual void start()
    {
        if (original_request->operation == Communication::Operation::BARRIER)
        {
            check_mpi(MPI_Ibarrier(original_request->comm, &mpi_request));
        }
        else
        {
            check_mpi(MPI_Start(&mpi_request));
        }
    }

    virtual bool done()
    {
        int value = 0;
        check_mpi(MPI_Test(&mpi_request, &value, MPI_STATUS_IGNORE));
        return value;
    }

protected:
    MPI_Request              mpi_request      = MPI_REQUEST_NULL;
    std::shared_ptr<Request> original_request = nullptr;
};

#endif