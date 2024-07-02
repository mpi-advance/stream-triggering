#include "abstract/queue.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIX_ST_Queue_entry_free_all(int len, MPIX_ST_Queue_entry queue_entries[])
{
	for(int index = 0; index < len; ++index)
	{
		QueueEntry *qe = (QueueEntry*) queue_entries[index];
        delete qe;
        queue_entries[index] = MPIX_ST_Queue_ENTRY_NULL;
    }

	return MPIX_SUCCESS;
}
}