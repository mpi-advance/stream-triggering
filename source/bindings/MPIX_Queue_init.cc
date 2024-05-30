#include "abstract/queue.hpp"
#include "queues/ThreadQueue.h"
#include "stream-triggering.h"

extern "C" {

int MPIX_Queue_init(MPIX_Queue *queue, MPIX_Queue_type type)
{
	Queue *the_queue;
	switch(type)
	{
		case THREAD:
			the_queue = new ThreadQueue();
			break;
		default:
			throw std::runtime_error("Queue type not enabled");
	}
	*queue = (MPIX_Queue) the_queue;
    
	return MPIX_SUCCESS;
}
}