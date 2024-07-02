#include "cuda.h"
#include "mpi.h"
#include "stream-triggering.h"

#include <iostream>
#include <unistd.h>

#define check_cuda(function)                                                                       \
	{                                                                                              \
		auto err = function;                                                                       \
		check_cuda_error(err, __FILE__, __LINE__);                                                 \
	}

#define force_cuda(function)                                                                       \
	{                                                                                              \
		auto err = function;                                                                       \
		check_cuda_error<true>(err, __FILE__, __LINE__);                                           \
	}

template<bool shouldThrow = false>
void check_cuda_error(const cudaError_t err, const char *filename, const int line)
{
	if(err != cudaSuccess)
	{
		std::cout << "(" << err << ") in " << filename << " on line " << line << " : "
		          << cudaGetErrorString(err) << std::endl;
		if constexpr(shouldThrow)
		{
			throw std::runtime_error(cudaGetErrorString(err));
		}
	}
}

const int num_iters = 1;

__global__ void init_buffers(int *send_buf, int *recv_buf)
{
	printf("Init the buffers!\n");
	*send_buf = 0;
	*recv_buf = -1;
}

__global__ void thread_process_iteration(int iter, int *send_buf, int *recv_buf)
{
	if(threadIdx.x > 1)
	{
		return;
	}

	if(*recv_buf != iter)
	{
		printf("Expected: %d Got: %d\n", *recv_buf, iter);
	}
	printf("Just finished iteration: %d \n", iter);
	(*send_buf)++;
}

int main()
{
	check_cuda(cudaSetDevice(0));

	int mode;
	MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mode);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Make CUDA Buffers
	void *send_buf = nullptr;
	force_cuda(cudaMalloc(&send_buf, sizeof(int)));
	void *recv_buf = nullptr;
	force_cuda(cudaMalloc(&recv_buf, sizeof(int)));
	init_buffers<<<1, 1>>>((int *) send_buf, (int *) recv_buf);
	check_cuda(cudaDeviceSynchronize());

	cudaStream_t my_stream;
	check_cuda(cudaStreamCreateWithFlags(&my_stream, cudaStreamNonBlocking));

	// Make requests
	MPI_Request my_reqs[2];
	if(0 == rank % 2)
	{
		MPI_Send_init(send_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &my_reqs[0]);
		MPI_Recv_init(recv_buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &my_reqs[1]);
	}
	else
	{
		MPI_Recv_init(recv_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &my_reqs[0]);
		MPI_Send_init(recv_buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &my_reqs[1]);
	}

	// Make queue and queue entries
	MPIX_ST_Queue my_queue;
	MPIX_ST_Queue_init(&my_queue, CUDA, &my_stream);

	MPIX_ST_Queue_entry my_entries[2];
	MPIX_Prepare_all(2, my_reqs, my_queue, my_entries);

	for(int i = 0; i < num_iters; ++i)
	{
		MPIX_Enqueue_entry(my_queue, my_entries[0]);
		std::cout << "A" << std::endl;
		MPIX_Enqueue_entry(my_queue, my_entries[1]);
		std::cout << "B" << std::endl;
		MPIX_Enqueue_waitall(my_queue);
		std::cout << "C" << std::endl;
		thread_process_iteration<<<1, 1, 0, my_stream>>>(i, (int *) send_buf, (int *) recv_buf);
		std::cout << "D" << std::endl;
		check_cuda(cudaPeekAtLastError());
	}

	std::cout << "Waiting on queue!" << std::endl;
	MPIX_ST_Queue_host_wait(my_queue);

	// Cleanup
	MPIX_ST_Queue_entry_free_all(2, my_entries);

	MPI_Request_free(&my_reqs[0]);
	MPI_Request_free(&my_reqs[1]);

	check_cuda(cudaFree(send_buf));
	check_cuda(cudaFree(recv_buf));

	MPIX_ST_Queue_free(&my_queue);

	MPI_Finalize();
}