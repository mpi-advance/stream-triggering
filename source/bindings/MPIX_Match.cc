#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Match(MPIS_Request request)
{
    using namespace Communication;
    std::shared_ptr<Request> the_request = convert_request(request);
    Queue*                   the_queue   = (Queue*)(ACTIVE_QUEUE);
    the_queue->match(the_request);

    return MPIS_SUCCESS;
}
}