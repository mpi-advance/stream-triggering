#include "helpers.hpp"

extern "C" {

int MPIS_Wait(MPIS_Request* request, MPI_Status* status)
{
    using namespace Communication;

    /* Early exit */
    if (*request == MPIS_REQUEST_NULL)
    {
        return MPIS_SUCCESS;
    }

    /* Can only wait on "match" requests ATM */
    if (RequestState::ONGOING != (*request)->state)
    {
        throw MPISException(MPIS_UNSUPPORTED_BEHAVIOR,
                            "MPIS_Wait can't wait on communicaiton requests yet");
    }

    std::shared_ptr<Request>* internal_request = convert_request_ptr(request);
    /* TODO: Give the user a status object back*/
    (*internal_request)->wait_on_match();

    /* Delete allocated MPIS_Request object */
    delete* request;
    /* Set it back to null */
    *request = MPIS_REQUEST_NULL;

    return MPIS_SUCCESS;
}
}