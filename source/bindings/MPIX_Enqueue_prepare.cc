#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Enqueue_prepare(MPIS_Queue queue, MPIS_Request request)
{
    using namespace Communication;
    Queue*   the_queue   = (Queue*)(queue);
    Request* the_request = (Request*)(request);

    the_queue->prepare(the_request);

    return MPIS_SUCCESS;
}
}