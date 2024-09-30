#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Enqueue_prepare(MPIS_Queue queue, MPIS_Request request)
{
    using namespace Communication;
    Queue*                   the_queue   = (Queue*)(queue);
    std::shared_ptr<Request> the_request = convert_request(request);

    the_queue->enqueue_prepare(the_request);

    return MPIS_SUCCESS;
}
}