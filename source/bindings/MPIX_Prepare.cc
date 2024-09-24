#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Prepare(MPIS_Request request)
{
    using namespace Communication;
	Request *the_request = (Request *) (request);
    the_request->prepare();
	return MPIS_SUCCESS;
}
}