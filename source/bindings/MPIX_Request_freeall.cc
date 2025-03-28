#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Request_freeall(int len, MPIS_Request requests[])
{
	for(int i = 0; i < len; ++i)
	{
		int err_code = MPIS_Request_free(&requests[i]);
		if (MPIS_SUCCESS != err_code)
			return err_code;
	}
	return MPIS_SUCCESS;
}
}