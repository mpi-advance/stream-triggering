#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Queue_wait(MPIS_Queue queue)
{
    MPIS_BINDING_ENTER

    Queue* the_queue = (Queue*)(queue);
    the_queue->host_wait();

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}