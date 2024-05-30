#include "mpi.h"
#include "stream-triggering.h"

#include <iostream>

const int num_iters = 100;
int      *send_bufs;
int       send_buf = -1;
int      *recv_bufs;
int       recv_buf = -1;

void thread_prepare_send(int iter)
{
	send_buf = send_bufs[iter];
}

void thread_process_recv(int iter)
{
	recv_bufs[iter] = recv_buf;
}

int main()
{
	MPI_Init(nullptr, nullptr);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	send_bufs = new int[num_iters];
	recv_bufs = new int[num_iters];

	for(int i = 0; i < num_iters; ++i)
	{
		send_bufs[i] = (rank * 100) + i;
		recv_bufs[i] = -1;
	}

	MPIX_Queue my_queue;
	MPIX_Queue_init(&my_queue, THREAD);

	MPI_Request my_reqs[2];
	if(0 == rank % 2)
	{
		MPI_Send_init(&send_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &my_reqs[0]);
		MPI_Recv_init(&recv_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &my_reqs[1]);
	}
	else
	{
		MPI_Recv_init(&recv_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &my_reqs[0]);
		MPI_Send_init(&send_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &my_reqs[1]);
	}

	thread_prepare_send(0);

	MPIX_Queue_entry my_entries[2];
	MPIX_Prepare_all(2, my_reqs, my_queue, my_entries);

	for(int i = 0; i < num_iters; ++i)
	{
		thread_prepare_send(i);
		MPIX_Enqueue_entry(my_queue, my_entries[0]);
		MPIX_Enqueue_entry(my_queue, my_entries[1]);
		MPIX_Enqueue_waitall(my_queue);
		MPIX_Queue_host_wait(my_queue); // Needed for just this example w/ CPU thread
		thread_process_recv(i);
	}

	for(int i = 0; i < num_iters; ++i)
	{
		if(0 == (rank % 2))
		{
			if(recv_bufs[i] != 100 + i)
				std::cout << "Expected: " << 100 + i << " Got: " << recv_bufs[i] << std::endl;
		}
		else
		{
			if(recv_bufs[i] != i)
				std::cout << "Expected: " << i << " Got: " << recv_bufs[i] << std::endl;
		}
	}

	MPIX_Queue_entry_free_all(2, my_entries);

	MPI_Request_free(&my_reqs[0]);
	MPI_Request_free(&my_reqs[1]);

	MPIX_Queue_free(&my_queue);

	delete[] send_bufs;
	delete[] recv_bufs;

	MPI_Finalize();
}