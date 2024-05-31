#include "mpi.h"
#include "stream-triggering.h"

#include <iostream>

const int num_iters = 100;
int       send_buf = -1;
int       recv_buf = -1;

void thread_process_iteration(int iter)
{
	if(recv_buf != iter)
	{
		std::cout << "Expected: " << iter << " Got: " << recv_buf << std::endl;
	}
	send_buf++;
}

int main()
{
	MPI_Init(nullptr, nullptr);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Make requests
	MPI_Request my_reqs[2];
	if(0 == rank % 2)
	{
		MPI_Send_init(&send_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &my_reqs[0]);
		MPI_Recv_init(&recv_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &my_reqs[1]);
	}
	else
	{
		MPI_Recv_init(&recv_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &my_reqs[0]);
		MPI_Send_init(&recv_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &my_reqs[1]);
	}

    // Prepare inital buffers
	send_buf = 0;
	recv_buf = -1;

    // Make queue and queue entries
    MPIX_Queue my_queue;
	MPIX_Queue_init(&my_queue, THREAD_SERIALIZED);
	MPIX_Queue_entry my_entries[2];
	MPIX_Prepare_all(2, my_reqs, my_queue, my_entries);

	for(int i = 0; i < num_iters; ++i)
	{
		MPIX_Enqueue_entry(my_queue, my_entries[0]);
		MPIX_Enqueue_entry(my_queue, my_entries[1]);
		MPIX_Enqueue_waitall(my_queue);
        // This next line is needed for the CPU Thread example because I can't just
        // "queue" a buffer change to the hidden thread like I can to a GPU thread
        MPIX_Queue_host_wait(my_queue);
		thread_process_iteration(i);
	}

    // Cleanup
	MPIX_Queue_entry_free_all(2, my_entries);

	MPI_Request_free(&my_reqs[0]);
	MPI_Request_free(&my_reqs[1]);

	MPIX_Queue_free(&my_queue);

	MPI_Finalize();
}