#include "abstract/queue.hpp"
#include "queues/ThreadQueue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIX_Queue_init(MPIX_Queue *queue, MPIX_Queue_type type)
{
	Queue *the_queue;
	switch(type)
	{
		case THREAD:
			the_queue = new ThreadQueue<false>();
			break;
		case THREAD_SERIALIZED:
			the_queue = new ThreadQueue<true>();
			break;
		default:
			throw std::runtime_error("Queue type not enabled");
	}
	*queue = (MPIX_Queue) the_queue;
    
	return MPIX_SUCCESS;
}
}