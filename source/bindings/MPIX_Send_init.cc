#include "helpers.hpp"

extern "C" {

int MPIS_Send_init(const void* buffer, MPI_Count count, MPI_Datatype datatype, int dest,
                   int tag, MPI_Comm comm, MPI_Info info, MPIS_Request* request)
{
    using namespace Communication;
    std::shared_ptr<Request>* internal_request = new std::shared_ptr<Request>(
        new Request(Operation::SEND, const_cast<void*>(buffer), count, datatype, dest,
                    tag, comm, info));

    *request =
        new MPIS_Request_struct{RequestState::UNMATCHED, (uintptr_t)internal_request};

    return MPIS_SUCCESS;
}
}