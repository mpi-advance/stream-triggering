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
            force_mpi(MPI_Recv_init(&prepare_request_buffer, 1, MPI_INT,
                                    req->peer, THREAD_PREPARE_TAG, req->comm,
                                    &prepare_request));
            break;
        case Communication::Operation::RECV:
            check_mpi(MPI_Recv_init(req->buffer, req->count, req->datatype,
                                    req->peer, req->tag, req->comm,
                                    &mpi_request));
            prepare_request_buffer = req->getID();
            force_mpi(MPI_Send_init(&prepare_request_buffer, 1, MPI_INT,
                                    req->peer, THREAD_PREPARE_TAG, req->comm,
                                    &prepare_request));
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

void ThreadRequest::prepare()
{
    check_mpi(MPI_Start(&prepare_request));
    check_mpi(MPI_Wait(&prepare_request, MPI_STATUS_IGNORE));
    original_request.lock()->ready();
}

bool ThreadRequest::canGo()
{
    return false;  // original_request->is_ready();
}

bool ThreadRequest::done()
{
    int value = 0;
    check_mpi(MPI_Test(&mpi_request, &value, MPI_STATUS_IGNORE));
    return value;
}