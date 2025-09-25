#include "abstract/queue.hpp"
#include "helpers.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Queue_free(MPIS_Queue* queue)
{
    Queue* the_queue = (Queue*)(*queue);
    delete the_queue;

    *queue = MPIS_QUEUE_NULL;

    ACTIVE_QUEUE = MPIS_QUEUE_NULL;

    return MPIS_SUCCESS;
}
}