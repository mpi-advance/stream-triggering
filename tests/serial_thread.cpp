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
	MPIS_Request my_reqs[2];
	if(0 == rank % 2)
	{
		MPIS_Send_init(&send_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &my_reqs[0]);
		MPIS_Recv_init(&recv_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &my_reqs[1]);
	}
	else
	{
		MPIS_Recv_init(&recv_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &my_reqs[0]);
		MPIS_Send_init(&recv_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &my_reqs[1]);
	}

    // Prepare inital buffers
	send_buf = 0;
	recv_buf = -1;

    // Make queue and queue entries
    MPIS_Queue my_queue;
	MPIS_Queue_init(&my_queue, THREAD_SERIALIZED, nullptr);

	for(int i = 0; i < num_iters; ++i)
	{
		MPIS_Enqueue_start(my_queue, my_reqs[0]);
		MPIS_Enqueue_start(my_queue, my_reqs[1]);
		MPIS_Enqueue_waitall(my_queue);
        // This next line is needed for the CPU Thread example because I can't just
        // "queue" a buffer change to the hidden thread like I can to a GPU thread
        MPIS_Queue_wait(my_queue);
		thread_process_iteration(i);
	}

	std::cout << "Finished all computation and communication!" << std::endl;

    // Cleanup
	MPIS_Request_free(&my_reqs[0]);
	MPIS_Request_free(&my_reqs[1]);

	MPIS_Queue_free(&my_queue);

	MPI_Finalize();
}