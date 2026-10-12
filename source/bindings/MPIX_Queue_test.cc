#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Queue_test(MPIS_Queue* queue, int* flag)
{
    MPIS_BINDING_ENTER

    Queue* the_queue = (Queue*)(*queue);
    *flag            = the_queue->queue_test() ? 1 : 0;

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}
