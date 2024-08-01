#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Prepare_all(int len, MPIS_Request requests[])
{
	return MPIS_SUCCESS;
}
}