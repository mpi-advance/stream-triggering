#include "helpers.hpp"

extern "C" {

int MPIS_Is_matched(MPIS_Request* request, int* matched)
{
    MPIS_BINDING_ENTER

    using namespace Communication;
    /* A match request is never itself matched */
    (*matched) =
        !(*request)->match_handle && (*convert_request_ptr(request))->is_matched();

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}