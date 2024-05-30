#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIX_Queue_free(MPIX_Queue *queue)
{
	Queue *the_queue = (Queue *) (*queue);
	delete the_queue;

	*queue = MPIX_QUEUE_NULL;
	return MPIX_SUCCESS;
}
}