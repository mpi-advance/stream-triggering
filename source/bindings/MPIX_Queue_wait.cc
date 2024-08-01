#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Queue_wait(MPIS_Queue queue)
{
	Queue *the_queue = (Queue *) (queue);
	the_queue->host_wait();
	return MPIS_SUCCESS;
}
}