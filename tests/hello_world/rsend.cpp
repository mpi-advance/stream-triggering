#include "../common/common.hpp"

int main(int argc, char* argv[])
{
    int mode;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Input parameters
    int num_warmups = 10;
    int BUFFER_SIZE = 64;

    // Info hint for buffer
    MPI_Info mem_info;
    MPI_Info_create(&mem_info);
#ifndef FINE_GRAINED_TEST
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:coarse");
#else
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:fine");
#endif

    // Make Buffers
    int BLOCK_SIZE = 128;
    int NUM_BLOCKS = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE * 2, mem_info, &send_buf);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE * 2, mem_info, &recv_buf);

    init_buffers3<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buf, (int*)recv_buf,
                                              BUFFER_SIZE * 2);
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

    // Make requests
    MPIS_Request my_reqs[2];
    int          offset = sizeof(int) * BUFFER_SIZE;
    if (0 == rank % 2)
    {
        MPIS_Rsend_init(send_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, mem_info,
                        &my_reqs[0]);
        MPIS_Send_init((char*)send_buf + offset, BUFFER_SIZE, MPI_INT, 1, 0,
                       MPI_COMM_WORLD, mem_info, &my_reqs[1]);
    }
    else
    {
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[0]);
        MPIS_Recv_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, 0, 0,
                       MPI_COMM_WORLD, mem_info, &my_reqs[1]);
    }

    MPIS_Matchall(2, my_reqs, MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << rank << " starting test!" << std::endl;
    double start = MPI_Wtime();
    if (1 == rank)
    {
        sleep(5);
    }
    MPIS_Enqueue_start(my_queue, &my_reqs[0]);
    MPIS_Enqueue_waitall(my_queue);
    MPIS_Queue_wait(my_queue);
    double end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << rank << " time A: " << end - start << std::endl;

    start = MPI_Wtime();
    if (1 == rank)
    {
        sleep(5);
    }
    MPIS_Enqueue_start(my_queue, &my_reqs[1]);
    MPIS_Enqueue_waitall(my_queue);

    MPIS_Queue_wait(my_queue);
    end = MPI_Wtime();

    std::cout << rank << " time B: " << end - start << std::endl;

    // Final check

    device_sync();
    if (1 == rank)
    {
        print_buffer3<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)recv_buf,
                                                                BUFFER_SIZE * 2, rank);
    }
    device_sync();

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    MPIS_Free_mem(send_buf);
    MPIS_Free_mem(recv_buf);

    MPI_Info_free(&mem_info);

    MPIS_Queue_free(&my_queue);

    std::cout << rank << " is done." << std::endl;

    MPI_Finalize();

    return 0;
}