#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Match(MPIS_Request request)
{
    using namespace Communication;
    Request* the_request = (Request*)(request);
    the_request->match();

    return MPIS_SUCCESS;
}
}