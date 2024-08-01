#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Request_free(MPIS_Request *req)
{
    using namespace Communication;
	Request *the_req = (Request *) (*req);
	delete the_req;

	*req = MPIS_REQUEST_NULL;
	return MPIS_SUCCESS;
}
}