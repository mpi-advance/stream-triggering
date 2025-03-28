#include <hip/hip_runtime.h>
#include <unistd.h>

#include <iostream>

#include "mpi.h"
#include "stream-triggering.h"

#define check_hip(function)                       \
    {                                             \
        auto err = function;                      \
        check_hip_error(err, __FILE__, __LINE__); \
    }

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

__global__ void init_buffers(int* send_buf, int* recv_buf, int buffer_len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    send_buf[index] = 0;
    recv_buf[index] = -1;
}

__global__ void pack_buffer(int* buffer, int buffer_len, int iteration)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    buffer[index] = iteration * 100;
}

__global__ void pack_buffer2(int* buffer, int* recvd_buffer, int buffer_len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    buffer[index] = recvd_buffer[index];
}

__global__ void print_buffer(volatile int* buffer, int buffer_len,
                             int iteration, int rank)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    if (buffer[index] != iteration * 100)
    {
        printf("<GPU %d> Wrong buffer value @ index: %d Got: %d Expected: %d\n",
               rank, index, buffer[index], iteration * 100);
    }
}

int main()
{
    const int num_iters = 1000;

    int mode;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mode);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Make CUDA Buffers
    int BUFFER_SIZE = 10;
    int BLOCK_SIZE  = 128;
    int NUM_BLOCKS  = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    void*  send_buf = nullptr;
    size_t buf_size = sizeof(int) * BUFFER_SIZE;
    force_hip(hipMalloc(&send_buf, buf_size));
    // force_hip(
    //    hipExtMallocWithFlags(&send_buf, buf_size, hipDeviceMallocFinegrained));
    void* recv_buf = nullptr;
    force_hip(hipMalloc(&recv_buf, buf_size));
    // force_hip(
    //    hipExtMallocWithFlags(&recv_buf, buf_size, hipDeviceMallocFinegrained));
    init_buffers<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buf, (int*)recv_buf,
                                             BUFFER_SIZE);

    check_hip(hipDeviceSynchronize());

    hipStream_t my_stream;
    check_hip(hipStreamCreateWithFlags(&my_stream, hipStreamNonBlocking));

    // Make queue
    MPIS_Queue my_queue;
    MPIS_Queue_init(&my_queue, CXI, &my_stream);

#define SEND_REQ (rank ^ 1)
#define RECV_REQ (rank & 1)

    // Info hint
    MPI_Info mem_info;
    MPI_Info_create(&mem_info);
    MPI_Info_set(mem_info, "MPIS_GPU_MEM_TYPE", "COARSE");
    // MPI_Info_set(mem_info, "MPIS_GPU_MEM_TYPE", "FINE");

    // Make requests
    MPIS_Request my_reqs[2];
    // MPIS_Request queue_reqs[2]; TODO
    if (0 == rank % 2)
    {
        MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD,
                       mem_info, &my_reqs[SEND_REQ]);
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD,
                       mem_info, &my_reqs[RECV_REQ]);
    }
    else
    {
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD,
                       mem_info, &my_reqs[RECV_REQ]);
        MPIS_Send_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD,
                       mem_info, &my_reqs[SEND_REQ]);
    }

    MPIS_Match(my_reqs[0]);
    MPIS_Match(my_reqs[1]);

    if (1 == rank)
    {
        MPIS_Enqueue_start(my_queue, my_reqs[RECV_REQ]);
    }
    MPIS_Queue_wait(my_queue);
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << rank << " After barrier!" << std::endl;

    for (int i = 0; i < num_iters; i++)
    {
        if (0 == rank)
        {
            // Ping side
            pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)send_buf, BUFFER_SIZE, i);
            MPIS_Enqueue_startall(my_queue, 2, my_reqs);
            MPIS_Enqueue_waitall(my_queue);
            print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)recv_buf, BUFFER_SIZE, i, rank);
        }
        else
        {
            MPIS_Enqueue_waitall(my_queue);
            print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)recv_buf, BUFFER_SIZE, i, rank);
            pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)send_buf, (int*)recv_buf, BUFFER_SIZE);
            // Pong
            if (i + 1 < num_iters)
            {
                MPIS_Enqueue_startall(my_queue, 2, my_reqs);
            }
            else
            {
                MPIS_Enqueue_start(my_queue, my_reqs[SEND_REQ]);
                MPIS_Enqueue_waitall(my_queue);
            }
        }

        //std::cout << rank << " After iter " << i << std::endl;
    }

    MPIS_Queue_wait(my_queue);

    std::cout << rank << " After all work!" << std::endl;
    // Final check
    check_hip(hipDeviceSynchronize());
    print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
        (int*)recv_buf, BUFFER_SIZE, num_iters - 1, rank);

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);

    MPI_Info_free(&mem_info);

    MPIS_Queue_free(&my_queue);

    std::cout << rank << " is done!" << std::endl;

    MPI_Finalize();
}