#include "../common/common.hpp"
#include "../common/timers.hpp"
#include "stream-triggering.h"

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
    Timing::init_timers(num_iters);

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

    void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &send_buffer);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &recv_buffer);

    init_buffers<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buffer, (int*)recv_buffer,
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

#define SEND_REQ (rank ^ 1)
#define RECV_REQ (rank & 1)

    // Make requests
    MPIS_Request my_reqs[2];
    if (0 == rank % 2)
    {
        MPIS_Send_init(send_buffer, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[SEND_REQ]);
        MPIS_Recv_init(recv_buffer, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[RECV_REQ]);
    }
    else
    {
        MPIS_Recv_init(recv_buffer, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[RECV_REQ]);
        MPIS_Send_init(send_buffer, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[SEND_REQ]);
    }

    MPIS_Match(&my_reqs[0], MPI_STATUS_IGNORE);
    MPIS_Match(&my_reqs[1], MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    auto do_cycles = [&]<bool TIMERS>(int num_cycles) {
        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == rank)
            {
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(my_queue);
#endif
                // Ping side
                pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)send_buffer,
                                                                      BUFFER_SIZE, i);
#ifdef THREAD_BACKEND
                device_sync();
#endif
                MPIS_Enqueue_startall(my_queue, 2, my_reqs);
                MPIS_Enqueue_waitall(my_queue);

                // #ifdef THREAD_BACKEND
                //                 MPIS_Queue_wait(my_queue);
                // #endif
                print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                    (int*)recv_buffer, BUFFER_SIZE, i, rank);
            }
            else
            {
                MPIS_Enqueue_start(my_queue, &my_reqs[RECV_REQ]);
                MPIS_Enqueue_waitall(my_queue);
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(my_queue);
#endif
                print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                    (int*)recv_buffer, BUFFER_SIZE, i, rank);
                pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                    (int*)send_buffer, (int*)recv_buffer, BUFFER_SIZE);
#ifdef THREAD_BACKEND
                device_sync();
#endif
                MPIS_Enqueue_start(my_queue, &my_reqs[SEND_REQ]);
                MPIS_Enqueue_waitall(my_queue);
            }

            if constexpr (TIMERS)
            {
                Timing::add_timer(i);
            }
        }

        MPIS_Queue_wait(my_queue);
    };

    do_cycles.template operator()<false>(num_warmups);
    MPI_Barrier(MPI_COMM_WORLD);
    Timing::set_base_timer();
    double             start = MPI_Wtime();
    do_cycles.template operator()<false>(num_iters);
    double             end = MPI_Wtime();

    // Final check
    device_sync();
    print_buffer<<<1, BLOCK_SIZE, 0, my_stream>>>((int*)recv_buffer, BUFFER_SIZE,
                                                  num_iters - 1, rank);
    device_sync();

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    MPIS_Free_mem(send_buffer);
    MPIS_Free_mem(recv_buffer);

    MPI_Info_free(&mem_info);

    MPIS_Queue_free(&my_queue);

    std::cout << rank << " is done: " << end - start << std::endl;
    // Timing::print_timers(rank);
    Timing::free_timers();

    MPI_Finalize();

    return 0;
}