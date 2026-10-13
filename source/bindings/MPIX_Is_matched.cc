#include "helpers.hpp"

extern "C" {

int MPIS_Is_matched(MPIS_Request* request, int* matched)
{
    MPIS_BINDING_ENTER

    using namespace Communication;
    std::shared_ptr<Request>* internal_request = convert_request_ptr(request);
    (*matched) = !(*request)->match_handle && (*internal_request)->is_matched();

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}