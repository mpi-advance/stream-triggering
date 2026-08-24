#include "helpers.hpp"

extern "C" {

int MPIS_Request_free(MPIS_Request* request)
{
    MPIS_BINDING_ENTER

    using namespace Communication;

    /* Early exit to avoid deleting memory we shouldn't */
    if (nullptr == request || MPIS_REQUEST_NULL == *request)
    {
        Print::out("Not freeing null request.");
        return MPIS_SUCCESS;
    }

    /* Match requests don't allocate anything, only normal requests do */
    if (RequestState::ONGOING != (*request)->state)
    {
        std::shared_ptr<Request>* internal_request = convert_request_ptr(request);
        /* Delete underlying request object allocated to
         * MPIS_Request_struct.internal_request */
        delete internal_request;
    }

    /* Delete allocated MPIS_Request object */
    delete *request;
    /* Set it back to null */
    *request = MPIS_REQUEST_NULL;

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}