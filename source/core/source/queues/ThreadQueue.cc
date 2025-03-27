#include "queues/ThreadQueue.hpp"

#include "safety/mpi.hpp"

ThreadRequest::ThreadRequest(std::shared_ptr<Request> req)
    : original_request(req)
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
        default:
            throw std::runtime_error("Invalid Request");
            break;
    }
}

void ThreadRequest::start()
{
    check_mpi(MPI_Start(&mpi_request));
}

bool ThreadRequest::done()
{
    int value = 0;
    check_mpi(MPI_Test(&mpi_request, &value, MPI_STATUS_IGNORE));
    return value;
}