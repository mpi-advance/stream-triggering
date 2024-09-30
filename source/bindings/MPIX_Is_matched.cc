#include "helpers.hpp"

extern "C" {

int MPIS_Is_matched(MPIS_Request request, int* matched)
{
    using namespace Communication;
    std::shared_ptr<Request> the_request = convert_request(request);
    (*matched)                           = the_request->is_matched();

    return MPIS_SUCCESS;
}
}