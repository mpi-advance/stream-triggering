#include <memory>

#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Barrier_init(MPI_Comm comm, MPI_Info info, MPIS_Request* request)
{
    using namespace Communication;
    std::shared_ptr<Request>* the_request =
        new std::shared_ptr<Request>(new Request(
            Operation::BARRIER, nullptr, 0, MPI_DATATYPE_NULL, -1, 0, comm, info));
    *request = (MPIS_Request)the_request;
    return MPIS_SUCCESS;
}
}