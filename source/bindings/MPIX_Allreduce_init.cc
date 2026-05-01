#include "helpers.hpp"

extern "C" {

int MPIS_Allreduce_init(const void* sendbuf, void* recvbuf, int count,
                        MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Info info,
                        MPIS_Request* request)
{
    using namespace Communication;
    std::shared_ptr<Request>* internal_request = new std::shared_ptr<Request>(
        new Request(Operation::ALLREDUCE, const_cast<void*>(sendbuf), recvbuf, count,
                    datatype, -1, 0, comm, info, op));

    *request =
        new MPIS_Request_struct{RequestState::UNMATCHED, (uintptr_t)internal_request};

    return MPIS_SUCCESS;
}
}