#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Ready(MPIS_Request request)
{
    using namespace Communication;
    Request* the_request = (Request*)(request);
    the_request->ready();
    return MPIS_SUCCESS;
}
}