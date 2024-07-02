#include "abstract/queue.hpp"
#include "stream-triggering.h"

#include <iostream>

extern "C" {

int MPIX_Prepare_all(int len, MPI_Request requests[], MPIX_ST_Queue queue, MPIX_ST_Queue_entry qentry[])
{
	Queue *the_queue = (Queue *) (queue);
	for(int index = 0; index < len; ++index)
	{
		QueueEntry *qe = the_queue->create_entry(requests[index]);
		qe->prepare();
		qentry[index]  = (MPIX_ST_Queue_entry) qe;
	}
	return MPIX_SUCCESS;
}
}