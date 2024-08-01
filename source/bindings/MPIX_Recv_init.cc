#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Recv_init(void         *buffer,
                   MPI_Count     count,
                   MPI_Datatype  datatype,
                   int           src,
                   int           tag,
                   MPI_Comm      comm,
                   MPI_Info      info,
                   MPIS_Request *request)
{
	using namespace Communication;
	Request *the_request =
	    new Request(Operation::RECV, buffer, count, datatype, src, tag, comm, info);
	*request = (MPIS_Request) the_request;
	return MPIS_SUCCESS;
}
}