#include "abstract/queue.hpp"
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

    Queue* the_queue = (Queue*)(ACTIVE_QUEUE);

    /* TODO: Give the user a status object back*/
    the_queue->finalize_match({*convert_request_ptr(request)});

    /* Delete allocated MPIS_Request object */
    delete *request;
    /* Set it back to null */
    *request = MPIS_REQUEST_NULL;

    return MPIS_SUCCESS;
}
}