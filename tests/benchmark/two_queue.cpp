#include "common.hpp"

int main(int argc, char* argv[])
{
    int mode;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // I want "two params"
    check_param_size(&argc, 2, "<program> <number of iterations> <buffer size>");

    // Input parameters
    int num_warmups = 10;
    int num_iters   = 0;
    int BUFFER_SIZE = 0;
    read_iter_buffer_input(&argv, &num_iters, &BUFFER_SIZE);

    // Make Buffers
    int BLOCK_SIZE = 128;
    int NUM_BLOCKS = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    allocate_gpu_memory(&send_buf, sizeof(int) * BUFFER_SIZE * 2);
    allocate_gpu_memory(&recv_buf, sizeof(int) * BUFFER_SIZE * 2);

    init_buffers<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buf, (int*)recv_buf,
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

    // Info hint
    MPI_Info mem_info;
    MPI_Info_create(&mem_info);
#ifndef FINE_GRAINED_TEST
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:coarse");
#else
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:fine");
#endif

#define SEND_REQ (rank ^ 1)
#define RECV_REQ (rank & 1)

    // Make requests
    MPIS_Request my_reqs[2];
    MPIS_Request my_other_reqs[2];
    // MPIS_Request queue_reqs[2]; TODO
    int offset = sizeof(int) * BUFFER_SIZE;
    if (0 == rank % 2)
    {
        MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[SEND_REQ]);
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[RECV_REQ]);
        MPIS_Send_init((char*)send_buf + offset, BUFFER_SIZE, MPI_INT, 1, 0,
                       MPI_COMM_WORLD, mem_info, &my_other_reqs[SEND_REQ]);
        MPIS_Recv_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, 1, 0,
                       MPI_COMM_WORLD, mem_info, &my_other_reqs[RECV_REQ]);
    }
    else
    {
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[RECV_REQ]);
        MPIS_Send_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[SEND_REQ]);
        MPIS_Recv_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, 0, 0,
                       MPI_COMM_WORLD, mem_info, &my_other_reqs[RECV_REQ]);
        MPIS_Send_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, 0, 0,
                       MPI_COMM_WORLD, mem_info, &my_other_reqs[SEND_REQ]);
    }

    MPIS_Match(&my_reqs[0], MPI_STATUS_IGNORE);
    MPIS_Match(&my_reqs[1], MPI_STATUS_IGNORE);
    MPIS_Match(&my_other_reqs[0], MPI_STATUS_IGNORE);
    MPIS_Match(&my_other_reqs[1], MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    void* active_send_buffer = send_buf;
    void* active_recv_buffer = recv_buf;

    void* inactive_send_buffer = (char*)send_buf + offset;
    void* inactive_recv_buffer = (char*)recv_buf + offset;

    MPIS_Request* active_request_ptr   = my_reqs;
    MPIS_Request* inactive_request_ptr = my_other_reqs;

    auto do_cycles = [&](int num_cycles) {
        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == rank)
            {
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(my_queue);
#endif
                // Ping side
                pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                    (int*)active_send_buffer, BUFFER_SIZE, i);
#ifdef THREAD_BACKEND
                check_gpu(hipDeviceSynchronize());
#endif
                MPIS_Enqueue_startall(my_queue, 2, active_request_ptr);
                MPIS_Enqueue_waitall(my_queue);
                print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                    (int*)active_recv_buffer, BUFFER_SIZE, i, rank);
            }
            else
            {
                MPIS_Enqueue_start(my_queue, &active_request_ptr[RECV_REQ]);
                MPIS_Enqueue_waitall(my_queue);
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(my_queue);
#endif
                print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                    (int*)active_recv_buffer, BUFFER_SIZE, i, rank);
                pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                    (int*)send_buf, (int*)active_recv_buffer, BUFFER_SIZE);
#ifdef THREAD_BACKEND
                check_gpu(hipDeviceSynchronize());
#endif
                MPIS_Enqueue_start(my_queue, &active_request_ptr[SEND_REQ]);
                MPIS_Enqueue_waitall(my_queue);
            }

            void* temp_send      = active_send_buffer;
            active_send_buffer   = inactive_send_buffer;
            inactive_send_buffer = temp_send;

            void* temp_recv      = active_recv_buffer;
            active_recv_buffer   = inactive_recv_buffer;
            inactive_recv_buffer = temp_recv;

            MPIS_Request* temp_reqs = active_request_ptr;
            active_request_ptr      = inactive_request_ptr;
            inactive_request_ptr    = temp_reqs;
        }

        MPIS_Queue_wait(my_queue);
    };

    do_cycles(num_warmups);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    do_cycles(num_iters);
    double end = MPI_Wtime();

    // Final check
    device_sync();
    print_buffer<<<1, BLOCK_SIZE, 0, my_stream>>>((int*)inactive_recv_buffer, BUFFER_SIZE,
                                                  num_iters - 1, rank);
    device_sync();

    // Cleanup of first queue
    MPIS_Request_freeall(2, my_reqs);
    MPIS_Request_freeall(2, my_other_reqs);
    MPIS_Queue_free(&my_queue);

    std::cout << rank << " is done: " << end - start << std::endl;

#if defined(HIP_BACKEND)
    MPIS_Queue_init(&my_queue, GPU_MEM_OPS, &my_stream);
#elif defined(CUDA_BACKEND)
    MPIS_Queue_init(&my_queue, GPU_MEM_OPS, &my_stream);
#elif defined(CXI_BACKEND)
    MPIS_Queue_init(&my_queue, CXI, &my_stream);
#elif defined(THREAD_BACKEND)
    MPIS_Queue_init(&my_queue, THREAD, &my_stream);
#endif

    if (0 == rank % 2)
    {
        MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[SEND_REQ]);
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[RECV_REQ]);
        MPIS_Send_init((char*)send_buf + offset, BUFFER_SIZE, MPI_INT, 1, 0,
                       MPI_COMM_WORLD, mem_info, &my_other_reqs[SEND_REQ]);
        MPIS_Recv_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, 1, 0,
                       MPI_COMM_WORLD, mem_info, &my_other_reqs[RECV_REQ]);
    }
    else
    {
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[RECV_REQ]);
        MPIS_Send_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[SEND_REQ]);
        MPIS_Recv_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, 0, 0,
                       MPI_COMM_WORLD, mem_info, &my_other_reqs[RECV_REQ]);
        MPIS_Send_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, 0, 0,
                       MPI_COMM_WORLD, mem_info, &my_other_reqs[SEND_REQ]);
    }

    std::cout << rank << " is done with second request creation" << std::endl;

    MPIS_Match(&my_reqs[0], MPI_STATUS_IGNORE);
    MPIS_Match(&my_reqs[1], MPI_STATUS_IGNORE);
    MPIS_Match(&my_other_reqs[0], MPI_STATUS_IGNORE);
    MPIS_Match(&my_other_reqs[1], MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << rank << " is done with second request match" << std::endl;

    active_send_buffer = send_buf;
    active_recv_buffer = recv_buf;

    inactive_send_buffer = (char*)send_buf + offset;
    inactive_recv_buffer = (char*)recv_buf + offset;

    active_request_ptr   = my_reqs;
    inactive_request_ptr = my_other_reqs;

    do_cycles(num_warmups);
    std::cout << rank << " is done with second warmup" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    do_cycles(num_iters);
    end = MPI_Wtime();

    std::cout << rank << " is done with second test" << std::endl;

    // Final check
    device_sync();
    print_buffer<<<1, BLOCK_SIZE, 0, my_stream>>>((int*)inactive_recv_buffer, BUFFER_SIZE,
                                                  num_iters - 1, rank);
    device_sync();

     std::cout << rank << " is done done: " << end - start << std::endl;

    // Cleanup of second time
    MPIS_Request_freeall(2, my_reqs);
    MPIS_Request_freeall(2, my_other_reqs);
    MPIS_Queue_free(&my_queue);
    MPI_Info_free(&mem_info);

    MPI_Finalize();

    return 0;
}