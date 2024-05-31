#include "queues/ThreadQueue.hpp"

#include "safety/mpi.hpp"

void ThreadQueueEntry::prepare()
{
	// Do nothing?
}

void ThreadQueueEntry::start()
{
	check_mpi(MPI_Start(&my_request));
}

bool ThreadQueueEntry::done()
{
	int value = 0;
	check_mpi(MPI_Test(&my_request, &value, MPI_STATUS_IGNORE));
	return value;
}

void ThreadQueueEntry::progress()
{
	done();
}