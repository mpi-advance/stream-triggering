#include "queues/HPEQueue.hpp"

#include "mpi.h"
#include "safety/hip.hpp"
#include "safety/mpi.hpp"

#include <hip/hip_runtime.h>

HPEQueueEntry::HPEQueueEntry(MPI_Request req) : QueueEntry(req) {}

HPEQueue::HPEQueue(hipStream_t *stream_addr) : Queue(), my_stream(stream_addr)
{
	force_mpi(MPI_Comm_dup(MPI_COMM_WORLD, &dup_comm));
	force_mpi(MPIS_Create_queue(dup_comm, (void *) my_stream, &my_queue));
}

HPEQueue::~HPEQueue()
{
	check_mpi(MPIS_Free_queue(my_queue));
	check_mpi(MPI_Comm_free(&dup_comm));
}

QueueEntry *HPEQueue::create_entry(MPI_Request the_request)
{
	return new HPEQueueEntry(the_request);
}

QueueEntry *HPEQueue::create_send(const void  *buffer,
                                  int          count,
                                  MPI_Datatype datatype,
                                  int          peer,
                                  int          tag)
{
	return new HPEQueueEntryP2P<true>(
	    const_cast<void *>(buffer), count, datatype, peer, tag, (void *) &my_queue);
}

QueueEntry *HPEQueue::create_recv(void *buffer, int count, MPI_Datatype datatype, int peer, int tag)
{
	return new HPEQueueEntryP2P<false>(buffer, count, datatype, peer, tag, (void *) &my_queue);
}

void HPEQueue::enqueue_operation(QueueEntry *qe)
{
	qe->start();
}

void HPEQueue::enqueue_waitall()
{
	force_mpi(MPIS_Enqueue_wait(my_queue));
}

void HPEQueue::host_wait()
{
    force_hip(hipStreamSynchronize(*my_stream));
}