#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Enqueue_start(MPIS_Queue queue, MPIS_Request* request)
{
    using namespace Communication;
    Queue* the_queue = (Queue*)(queue);

    std::shared_ptr<Request>* internal_request =
        convert_request_ptr(request, RequestState::MATCHED);

    the_queue->enqueue_operation(*internal_request);

    return MPIS_SUCCESS;
}
}