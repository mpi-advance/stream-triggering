#include "abstract/queue.hpp"
#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Queue_match(MPIS_Queue queue, MPIS_Request request)
{
    using namespace Communication;
    Request* the_request = (Request*)(request);
    Queue*   the_queue   = (Queue*)(queue);
    the_queue->match(the_request);

    return MPIS_SUCCESS;
}
}