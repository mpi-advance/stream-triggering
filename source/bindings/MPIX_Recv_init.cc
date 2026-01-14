#include "helpers.hpp"

extern "C" {

int MPIS_Recv_init(void* buffer, MPI_Count count, MPI_Datatype datatype, int src, int tag,
                   MPI_Comm comm, MPI_Info info, MPIS_Request* request)
{
    using namespace Communication;
    std::shared_ptr<Request>* internal_request = new std::shared_ptr<Request>(
        new Request(Operation::RECV, nullptr, buffer, count, datatype, src, tag, comm, info));

    *request =
        new MPIS_Request_struct{RequestState::UNMATCHED, (uintptr_t)internal_request};

    return MPIS_SUCCESS;
}
}