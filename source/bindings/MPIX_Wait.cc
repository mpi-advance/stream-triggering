#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Wait(MPIS_Request* request, MPI_Status* status)
{
    MPIS_BINDING_ENTER

    using namespace Communication;

    /* Early exit */
    if (nullptr == request || MPIS_REQUEST_NULL == *request)
    {
        return MPIS_SUCCESS;
    }

    Queue* the_queue = (Queue*)(ACTIVE_QUEUE);

    /* Can only wait on "match" requests ATM; TODO: Give the user a status object back*/
    the_queue->finalize_match({*convert_request_ptr(request, RequestState::ONGOING)});

    /* Delete allocated MPIS_Request object */
    delete *request;
    /* Set it back to null */
    *request = MPIS_REQUEST_NULL;

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}