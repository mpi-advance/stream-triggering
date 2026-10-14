#include "abstract/queue.hpp"
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

    if ((*request)->match_handle)
    {
        /* Match requests only own the list of Requests being matched */
        delete convert_match_request(*request);
    }
    else
    {
        /* A Request still being matched is in use by its match request */
        std::shared_ptr<Request>* internal_request = convert_request_ptr(
            request,
            {Request::State::UNMATCHED, Request::State::MATCHED, Request::State::ACTIVE});
        /* Tell the global queue that it needs to free resources related to this request*/
        Queue* the_queue = (Queue*)(ACTIVE_QUEUE);
        the_queue->request_free(*internal_request);
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