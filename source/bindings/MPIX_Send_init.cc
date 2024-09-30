#include <memory>

#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Send_init(const void* buffer, MPI_Count count, MPI_Datatype datatype,
                   int dest, int tag, MPI_Comm comm, MPI_Info info,
                   MPIS_Request* request)
{
    using namespace Communication;
    std::shared_ptr<Request>* the_request = new std::shared_ptr<Request>(
        new Request(Operation::SEND, const_cast<void*>(buffer), count, datatype,
                    dest, tag, comm, info));
    *request = (MPIS_Request)the_request;
    return MPIS_SUCCESS;
}
}