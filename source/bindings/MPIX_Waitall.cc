#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Waitall(int len, MPIS_Request requests[], MPI_Status statuses[])
{
    MPIS_BINDING_ENTER

    using namespace Communication;
    Queue*                                the_queue = (Queue*)(ACTIVE_QUEUE);
    std::vector<std::shared_ptr<Request>> internal_requests(len);

    Print::out("MPIS_Requests to wait on:", len);
    /* Convert to internal object */
    for (int i = 0; i < len; ++i)
    {
        MPIS_Request request = requests[i];

        if (MPIS_REQUEST_NULL == request)
        {
            continue;
        }

        internal_requests[i] = *convert_request(request, RequestState::ONGOING);
    }

    /* Do the wait all */
    the_queue->finalize_match(internal_requests);

    /* Now do cleanup */
    for (int i = 0; i < len; ++i)
    {
        MPIS_Request request = requests[i];

        if (MPIS_REQUEST_NULL == request)
        {
            continue;
        }

        /* Delete allocated MPIS_Request object */
        delete request;
        /* Set it back to null */
        requests[i] = MPIS_REQUEST_NULL;
    }

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}