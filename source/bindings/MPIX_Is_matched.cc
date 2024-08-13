#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Is_matched(MPIS_Request* request, int* matched)
{
    using namespace Communication;
	Request *the_request = (Request *) (*request);
    (*matched) = the_request->is_matched();

	return MPIS_SUCCESS;
}
}