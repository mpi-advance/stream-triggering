#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Enqueue_waitall(MPIS_Queue queue)
{
	Queue *the_queue = (Queue *) (queue);
	the_queue->enqueue_waitall();
	return MPIS_SUCCESS;
}
}