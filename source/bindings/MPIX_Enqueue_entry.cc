#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIX_Enqueue_entry(MPIX_Queue queue, MPIX_Queue_entry qentry)
{
	Queue      *the_queue       = (Queue *) (queue);
	QueueEntry *the_queue_entry = (QueueEntry *) (qentry);
	the_queue->enqueue_operation(the_queue_entry);
	return MPIX_SUCCESS;
}
}