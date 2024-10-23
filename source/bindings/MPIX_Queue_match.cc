#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Queue_match(MPIS_Queue queue, MPIS_Request request)
{
    using namespace Communication;
    std::shared_ptr<Request> the_request = convert_request(request);
    Queue*                   the_queue   = (Queue*)(queue);
    the_queue->match(the_request);

    return MPIS_SUCCESS;
}
}