#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Enqueue_waitall(MPIS_Queue queue)
{
	MPIS_BINDING_ENTER

	Queue *the_queue = (Queue *) (queue);
	the_queue->enqueue_waitall();

	MPIS_BINDING_EXIT
	return MPIS_SUCCESS;
}
}