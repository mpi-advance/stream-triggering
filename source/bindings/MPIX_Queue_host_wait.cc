#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIX_Queue_host_wait(MPIX_Queue queue)
{
	Queue *the_queue = (Queue *) (queue);
	the_queue->host_wait();
	return MPIX_SUCCESS;
}
}