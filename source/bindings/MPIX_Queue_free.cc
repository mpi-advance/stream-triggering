#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIX_ST_Queue_free(MPIX_ST_Queue *queue)
{
	Queue *the_queue = (Queue *) (*queue);
	delete the_queue;

	*queue = MPIX_ST_Queue_NULL;
	return MPIX_SUCCESS;
}
}