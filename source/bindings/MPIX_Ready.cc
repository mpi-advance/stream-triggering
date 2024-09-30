#include "helpers.hpp"

extern "C" {

int MPIS_Ready(MPIS_Request request)
{
    using namespace Communication;
    std::shared_ptr<Request> the_request = convert_request(request);
    the_request->ready();
    return MPIS_SUCCESS;
}
}