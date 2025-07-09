#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Enqueue_startall(MPIS_Queue queue, int len, MPIS_Request requests[])
{
    Queue*                                the_queue = (Queue*)(queue);
    std::vector<std::shared_ptr<Request>> all_requests(len);

    for (int i = 0; i < len; ++i)
    {
        all_requests[i] = convert_request(requests[i]);
    }

    the_queue->enqueue_startall(all_requests);

    return MPIS_SUCCESS;
}
}