#include "helpers.hpp"

extern "C" {

int MPIS_Is_matched(MPIS_Request* request, int* matched)
{
    using namespace Communication;
    std::shared_ptr<Request>* internal_request = convert_request_ptr(request);
    (*matched) =
        (*internal_request)->is_matched() && (RequestState::MATCHED == (*request)->state);

    return MPIS_SUCCESS;
}
}