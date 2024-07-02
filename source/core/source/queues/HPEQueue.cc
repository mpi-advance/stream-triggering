#include "queues/HPEQueue.hpp"

#include "safety/hip.hpp"
#include "safety/mpi.hpp"

#include <hip/hip_runtime.h>

#include "mpi.h"

HPEQueueEntry::HPEQueueEntry(MPI_Request req) : QueueEntry(req) {}

HPEQueue::HPEQueue(hipStream_t *stream_addr) : Queue(), my_stream(stream_addr)
{
    force_mpi(MPI_Comm_dup(MPI_COMM_WORLD, &dup_comm));
	force_mpi(MPIX_Create_queue(dup_comm, (void *) my_stream, &my_queue));
}

HPEQueue::~HPEQueue()
{
    check_mpi(MPIX_Free_queue(my_queue));
    check_mpi(MPI_Comm_free(&dup_comm));
}

QueueEntry *HPEQueue::create_entry(MPI_Request the_request)
{
	return new HPEQueueEntry(the_request);
}

void HPEQueue::enqueue_waitall() {}

void HPEQueue::host_wait() {}