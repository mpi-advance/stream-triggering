#include <hip/hip_runtime.h>

#include <iostream>

#include "mpi.h"
#include "stream-triggering.h"

#define force_hip(function)                             \
    {                                                   \
        auto err = function;                            \
        check_hip_error<true>(err, __FILE__, __LINE__); \
    }

template <bool shouldThrow = false>
void check_hip_error(const hipError_t err, const char* filename, const int line)
{
    if (err != hipSuccess)
    {
        std::cout << "(" << err << ") in " << filename << " on line " << line
                  << " : " << hipGetErrorString(err) << std::endl;
        if constexpr (shouldThrow)
        {
            throw std::runtime_error(hipGetErrorString(err));
        }
    }
}

__global__ void print_buffer(volatile int* buffer, int buffer_len,
                             int iteration, int rank)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len) return;

    if (buffer[index] != iteration * 100)
    {
        printf("<GPU %d> Wrong buffer value @ index: %d Got: %d Expected: %d\n",
               rank, index, buffer[index], iteration * 100);
    }
}

__global__ void set_buffer(int* buffer, int buffer_len, int value)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    buffer[index] = value;
}

__global__ void pack_buffer(int* buffer, int buffer_len, int iteration)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len) return;

    buffer[index] = iteration * 100;
}

__global__ void pack_buffer2(int* buffer, int* recvd_buffer, int buffer_len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len) return;

    buffer[index] = recvd_buffer[index];
}

int main()
{
    MPI_Init(nullptr, nullptr);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    constexpr int num_items  = 10;
    size_t        BLOCK_SIZE = 128;
    size_t        NUM_BLOCKS = (num_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t        total_size = sizeof(int) * num_items;
    void*         send_buffer;
    void*         recv_buffer;
    force_hip(hipMalloc(&send_buffer, total_size));
    force_hip(hipMalloc(&recv_buffer, total_size));

    // Prepare inital buffers
    set_buffer<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buffer, num_items, 0);
    set_buffer<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)recv_buffer, num_items, -1);
    force_hip(hipDeviceSynchronize());

    // Make requests
    MPIS_Request my_reqs[2];
    if (0 == rank % 2)
    {
        MPIS_Send_init(send_buffer, num_items, MPI_INT, 1, 0, MPI_COMM_WORLD,
                       MPI_INFO_NULL, &my_reqs[0]);
        MPIS_Recv_init(recv_buffer, num_items, MPI_INT, 1, 0, MPI_COMM_WORLD,
                       MPI_INFO_NULL, &my_reqs[1]);
    }
    else
    {
        MPIS_Recv_init(recv_buffer, num_items, MPI_INT, 0, 0, MPI_COMM_WORLD,
                       MPI_INFO_NULL, &my_reqs[0]);
        MPIS_Send_init(recv_buffer, num_items, MPI_INT, 0, 0, MPI_COMM_WORLD,
                       MPI_INFO_NULL, &my_reqs[1]);
    }

    hipStream_t stream;
    force_hip(hipStreamCreate(&stream));

    // Make queue and queue entries
    MPIS_Queue my_queue;
    MPIS_Queue_init(&my_queue, CXI, &stream);

    // Match
    MPIS_Queue_match(my_queue, my_reqs[0]);
    MPIS_Queue_match(my_queue, my_reqs[1]);
    std::cout << "Queue match" << std::endl;

    const int num_iters = 2;
    for (int i = 0; i < num_iters; ++i)
    {
        if (0 == rank % 2)
        {
            pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0 , stream>>>((int*)send_buffer,
                                                    num_items, i);
            MPIS_Enqueue_start(my_queue, my_reqs[0]);
            MPIS_Enqueue_start(my_queue, my_reqs[1]);
        }
        else
        {
            MPIS_Enqueue_start(my_queue, my_reqs[0]);
            MPIS_Enqueue_waitall(my_queue);
            pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0 , stream>>>(
                (int*)send_buffer, (int*)recv_buffer, num_items);
        }
        MPIS_Enqueue_waitall(my_queue);
    }
    std::cout << "Finished Queueing!" << std::endl;
    MPIS_Queue_wait(my_queue);
    std::cout << "Finished all computation and communication!" << std::endl;

    print_buffer<<<1, 16>>>((int*)recv_buffer, num_items,
                           num_iters - 1, rank);
    force_hip(hipDeviceSynchronize());

    // Cleanup
    MPIS_Request_free(&my_reqs[0]);
    MPIS_Request_free(&my_reqs[1]);

    MPIS_Queue_free(&my_queue);

    force_hip(hipStreamDestroy(stream));

    force_hip(hipFree(send_buffer));
    force_hip(hipFree(recv_buffer));

    MPI_Finalize();
}
