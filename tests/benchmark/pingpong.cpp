#include "common.hpp"

int main(int argc, char* argv[])
{
    int mode;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // I want "two params"
    check_param_size(&argc, 2,
                     "<program> <number of iterations> <buffer size>");

    // Input parameters
    int num_iters   = 0;
    int BUFFER_SIZE = 0;
    read_iter_buffer_input(&argv, &num_iters, &BUFFER_SIZE);

    // Make Buffers
    int BLOCK_SIZE = 128;
    int NUM_BLOCKS = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    allocate_gpu_memory(&send_buf, sizeof(int) * BUFFER_SIZE);
    allocate_gpu_memory(&recv_buf, sizeof(int) * BUFFER_SIZE);

    init_buffers<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buf, (int*)recv_buf,
                                             BUFFER_SIZE);
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

    // Info hint
    MPI_Info mem_info;
    MPI_Info_create(&mem_info);
#ifndef FINE_GRAINED_TEST
    MPI_Info_set(mem_info, "MPIS_GPU_MEM_TYPE", "COARSE");
#else
    MPI_Info_set(mem_info, "MPIS_GPU_MEM_TYPE", "FINE");
#endif

#define SEND_REQ (rank ^ 1)
#define RECV_REQ (rank & 1)

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

    MPIS_Request barrier_req;
    MPIS_Barrier_init(MPI_COMM_WORLD, MPI_INFO_NULL, &barrier_req);

    MPIS_Match(my_reqs[0]);
    MPIS_Match(my_reqs[1]);
    MPIS_Match(barrier_req);

    double start = MPI_Wtime();
    for (int i = 0; i < num_iters; i++)
    {
        if (0 == rank)
        {
#ifdef THREAD_BACKEND
            MPIS_Queue_wait(my_queue);
#endif
            // Ping side
            pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)send_buf, BUFFER_SIZE, i);
#ifdef THREAD_BACKEND
            check_gpu(hipDeviceSynchronize());
#endif
            MPIS_Enqueue_startall(my_queue, 2, my_reqs);
            MPIS_Enqueue_waitall(my_queue);
            // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
            //     (int*)recv_buf, BUFFER_SIZE, i, rank);
        }
        else
        {
            MPIS_Enqueue_start(my_queue, my_reqs[RECV_REQ]);
            MPIS_Enqueue_waitall(my_queue);
#ifdef THREAD_BACKEND
            MPIS_Queue_wait(my_queue);
#endif
            // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
            //     (int*)recv_buf, BUFFER_SIZE, i, rank);
            pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)send_buf, (int*)recv_buf, BUFFER_SIZE);
#ifdef THREAD_BACKEND
            check_gpu(hipDeviceSynchronize());
#endif
            MPIS_Enqueue_start(my_queue, my_reqs[SEND_REQ]);
            MPIS_Enqueue_waitall(my_queue);
        }

        MPIS_Enqueue_start(my_queue, barrier_req);
    }

    // std::cout << rank << " at final wait!" << std::endl;

    MPIS_Enqueue_waitall(my_queue);
    MPIS_Queue_wait(my_queue);
    double end = MPI_Wtime();

    // Final check
    device_sync();
    print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
        (int*)recv_buf, BUFFER_SIZE, num_iters - 1, rank);
    device_sync();

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    MPIS_Request_free(&barrier_req);

    MPI_Info_free(&mem_info);

    MPIS_Queue_free(&my_queue);

    std::cout << rank << " is done: " << end - start << std::endl;

    MPI_Finalize();

    return 0;
}