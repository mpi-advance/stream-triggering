#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIX_Enqueue_waitall(MPIX_ST_Queue queue)
{
	Queue *the_queue = (Queue *) (queue);
	the_queue->enqueue_waitall();
	return MPIX_SUCCESS;
}
}