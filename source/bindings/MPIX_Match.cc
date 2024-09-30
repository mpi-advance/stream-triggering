#include "helpers.hpp"

extern "C" {

int MPIS_Match(MPIS_Request request)
{
    using namespace Communication;
    std::shared_ptr<Request> the_request = convert_request(request);
    the_request->match();

    return MPIS_SUCCESS;
}
}