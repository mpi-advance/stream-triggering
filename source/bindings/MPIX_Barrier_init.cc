#include "helpers.hpp"

extern "C" {

int MPIS_Barrier_init(MPI_Comm comm, MPI_Info info, MPIS_Request* request)
{
    using namespace Communication;
    std::shared_ptr<Request>* internal_request = new std::shared_ptr<Request>(new Request(
        Operation::BARRIER, nullptr, nullptr, 0, MPI_DATATYPE_NULL, -1, 0, comm, info));

    *request =
        new MPIS_Request_struct{RequestState::UNMATCHED, (uintptr_t)internal_request};

    return MPIS_SUCCESS;
}
}