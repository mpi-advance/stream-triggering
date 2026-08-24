#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Enqueue_start(MPIS_Queue queue, MPIS_Request* request)
{
    MPIS_BINDING_ENTER
    using namespace Communication;
    Queue* the_queue = (Queue*)(queue);

    std::shared_ptr<Request>* internal_request =
        convert_request_ptr(request, RequestState::MATCHED);

    the_queue->enqueue_operation(*internal_request);

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}