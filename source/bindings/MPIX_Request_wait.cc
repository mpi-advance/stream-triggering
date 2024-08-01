#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Request_wait(MPIS_Request *req)
{
	return MPIS_SUCCESS;
}
}