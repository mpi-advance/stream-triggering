#include "abstract/queue.hpp"
#include "helpers.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Queue_free(MPIS_Queue* queue)
{
    MPIS_BINDING_ENTER

    if(nullptr == queue || MPIS_QUEUE_NULL == *queue)
    {
        Print::out("Not freeing null queue.");
        return MPI_SUCCESS;
    }

    Queue* the_queue = (Queue*)(*queue);
    delete the_queue;

    *queue = MPIS_QUEUE_NULL;

    ACTIVE_QUEUE = MPIS_QUEUE_NULL;

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}