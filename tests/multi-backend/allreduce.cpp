#include "../common/common.hpp"
#include "stream-triggering.h"

__global__ void verify(volatile int* buffer, int buffer_len, int rank, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    int expected = ((size*(size-1))/2)*100 + (size*index);

    if (buffer[index] != expected)
    {
        printf("<GPU %d> Wrong buffer value @ index: %d Got: %d Expected: %d\n", rank,
               index, buffer[index], expected);
    }
}

int main(int argc, char* argv[])
{
    int mode;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Input parameters
    int BUFFER_SIZE = 64;

    // Info hint for buffer
    MPI_Info mem_info;
    MPI_Info_create(&mem_info);
#ifdef NEED_HIP
#ifndef FINE_GRAINED_TEST
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:coarse");
#else
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:fine");
#endif
#elif defined(NEED_CUDA)
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "cuda:device");
#endif
    // Make Buffers
    int BLOCK_SIZE = 128;
    int NUM_BLOCKS = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &send_buf);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &recv_buf);

    init_buffers2<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buf, (int*)recv_buf, BUFFER_SIZE,
                                              rank);
    device_sync();

#if defined(NEED_HIP)
    hipStream_t my_stream;
    check_gpu(hipStreamCreateWithFlags(&my_stream, hipStreamNonBlocking));
#elif defined(NEED_CUDA)
    cudaStream_t my_stream;
    check_gpu(cudaStreamCreateWithFlags(&my_stream, cudaStreamNonBlocking));
#endif

    // Make queue
    MPIS_Queue my_queue;
#if defined(HIP_BACKEND)
    MPIS_Queue_init(&my_queue, GPU_MEM_OPS, &my_stream);
#elif defined(CUDA_BACKEND)
    MPIS_Queue_init(&my_queue, GPU_MEM_OPS, &my_stream);
#elif defined(CXI_BACKEND)
    MPIS_Queue_init(&my_queue, CXI, &my_stream);
#elif defined(THREAD_BACKEND)
    MPIS_Queue_init(&my_queue, THREAD, &my_stream);
#endif

    // Make request
    MPIS_Request my_req;
    MPIS_Allreduce_init(send_buf, recv_buf, BUFFER_SIZE, MPI_INT, MPI_SUM, MPI_COMM_WORLD, MPI_INFO_NULL, &my_req);
    MPIS_Queue_match(my_queue, &my_req, MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << rank << " starting test!" << std::endl;

    double start = MPI_Wtime();
    MPIS_Enqueue_start(my_queue, &my_req);
    MPIS_Enqueue_waitall(my_queue);
    MPIS_Queue_wait(my_queue);
    double end = MPI_Wtime();

    std::cout << rank << " done with GPU communication" << std::endl;
    std::cout << rank << " time: " << end - start << std::endl;

    // Final check
    device_sync();
    verify<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)recv_buf, BUFFER_SIZE,
                                                            rank, size);
    device_sync();

    // Cleanup
    MPIS_Request_freeall(1, &my_req);
    MPIS_Free_mem(send_buf);
    MPIS_Free_mem(recv_buf);

    MPI_Info_free(&mem_info);

    MPIS_Queue_free(&my_queue);

    std::cout << rank << " is done." << std::endl;

    MPI_Finalize();

    return 0;
}