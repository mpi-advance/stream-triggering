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
        : threshold(0), mpi_request(std::make_shared<MPI_Request>(MPI_REQUEST_NULL)), original_request(req)
    {
        MPI_Request* req_addr = mpi_request.get();
        switch (req->operation)
        {
            case Communication::Operation::SEND:
                check_mpi(MPI_Send_init(req->send_buffer, req->count, req->datatype,
                                        req->peer, req->tag, req->comm, req_addr));
                break;
            case Communication::Operation::RSEND:
                check_mpi(MPI_Rsend_init(req->send_buffer, req->count, req->datatype,
                                         req->peer, req->tag, req->comm, req_addr));
                break;
            case Communication::Operation::RECV:
                check_mpi(MPI_Recv_init(req->recv_buffer, req->count, req->datatype,
                                        req->peer, req->tag, req->comm, req_addr));
                break;
            case Communication::Operation::BARRIER:
                check_mpi(MPI_Barrier_init(req->comm, req->info, req_addr));
                break;
            case Communication::Operation::ALLREDUCE:
                check_mpi(MPI_Allreduce_init(req->send_buffer, req->recv_buffer,
                                             req->count, req->datatype, req->op,
                                             req->comm, req->info, req_addr));
                break;
            default:
                throw std::runtime_error("Invalid Request");
                break;
        }
        Print::always("Request made:", req->peer, req_addr);
    }

    virtual ~QueueEntry()
    {
        if (mpi_request)
        {
            check_mpi(MPI_Request_free(mpi_request.get()));
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
          start_location(other.start_location),
          wait_location(other.wait_location)
    {
        Print::out("MovedC:", other.mpi_request);
        // clear other structs
        other.threshold   = 0;
        other.mpi_request.reset();
        other.original_request.reset();
        other.start_location = nullptr;
        other.wait_location  = nullptr;
    }
    QueueEntry& operator=(QueueEntry&& other) noexcept
    {
        if (this != &other)
        {
            Print::out("MovedO:", other.mpi_request);
            threshold        = other.threshold;
            mpi_request      = other.mpi_request;
            original_request = other.original_request;
            start_location   = other.start_location;
            wait_location    = other.wait_location;
            // clear other
            other.threshold   = 0;
            other.mpi_request.reset();
            other.original_request.reset();
            other.start_location = nullptr;
            other.wait_location  = nullptr;
        }
        return *this;
    }

    operator std::shared_ptr<Progress::StartEntry>()
    {
        return std::make_shared<Progress::StartEntry>(mpi_request, start_location,
                                                      threshold);
    }

    operator std::shared_ptr<Progress::WaitEntry>()
    {
        return std::make_shared<Progress::WaitEntry>(mpi_request, wait_location,
                                                     threshold);
    }

    virtual void increment()
    {
        threshold++;
    }

    virtual Progress::CounterType get_threshold()
    {
        return threshold;
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

    volatile Progress::CounterType* get_start_location()
    {
        return start_location;
    }

    Progress::CounterType* get_wait_location()
    {
        return wait_location;
    }

    MPI_Request get_mpi_request()
    {
        return *mpi_request;
    }

protected:
    Progress::CounterType threshold = 0;

    std::shared_ptr<MPI_Request> mpi_request  = nullptr;
    std::shared_ptr<Request> original_request = nullptr;

    Progress::CounterType* start_location;
    Progress::CounterType* wait_location;
};

#endif