#include "../common/common.hpp"
#include "../common/timers.hpp"

void allocate_gpu_memory(void** location, size_t size)
{
#ifndef FINE_GRAINED_TEST
    force_gpu(hipMalloc(location, size));
#else
    force_gpu(hipExtMallocWithFlags(location, size, hipDeviceMallocFinegrained));
#endif
}

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

    // Make Buffers
    int BLOCK_SIZE = 128;
    int NUM_BLOCKS = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    allocate_gpu_memory(&send_buffer, sizeof(int) * BUFFER_SIZE);
    allocate_gpu_memory(&recv_buffer, sizeof(int) * BUFFER_SIZE);

    init_buffers<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buffer, (int*)recv_buffer, BUFFER_SIZE);
    device_sync();

#if defined(NEED_HIP)
    hipStream_t my_stream;
    check_gpu(hipStreamCreateWithFlags(&my_stream, hipStreamNonBlocking));
#elif defined(NEED_CUDA)
    cudaStream_t my_stream;
    check_gpu(cudaStreamCreateWithFlags(&my_stream, cudaStreamNonBlocking));
#endif

#define SEND_REQ (rank ^ 1)
#define RECV_REQ (rank & 1)

    // Make requests
    MPI_Request my_reqs[2];
    if (0 == rank % 2)
    {
        MPI_Send_init(send_buffer, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD,
                      &my_reqs[SEND_REQ]);
        MPI_Recv_init(recv_buffer, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD,
                      &my_reqs[RECV_REQ]);
    }
    else
    {
        MPI_Recv_init(recv_buffer, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD,
                      &my_reqs[RECV_REQ]);
        MPI_Send_init(send_buffer, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD,
                      &my_reqs[SEND_REQ]);
    }

    auto do_cycles = [&]<bool TIMERS>(int num_cycles) {
        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == rank)
            {
                // Ping side
                pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)send_buffer,
                                                                      BUFFER_SIZE, i);

                device_sync();
                MPI_Startall(2, my_reqs);
                MPI_Waitall(2, my_reqs, MPI_STATUSES_IGNORE);
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                //    (int*)recv_buffer, BUFFER_SIZE, i, rank);
            }
            else
            {
                MPI_Start(&my_reqs[RECV_REQ]);
                MPI_Wait(&my_reqs[RECV_REQ], MPI_STATUS_IGNORE);
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                //    (int*)recv_buffer, BUFFER_SIZE, i, rank);
                pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                    (int*)send_buffer, (int*)recv_buffer, BUFFER_SIZE);
                device_sync();
                MPI_Start(&my_reqs[SEND_REQ]);
                MPI_Wait(&my_reqs[SEND_REQ], MPI_STATUS_IGNORE);
            }

            if constexpr (TIMERS)
            {
                Timing::add_timer(i);
            }
        }
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
    MPI_Request_free(&my_reqs[SEND_REQ]);
    MPI_Request_free(&my_reqs[RECV_REQ]);

    std::cout << rank << " is done: " << end - start << std::endl;
    //Timing::print_timers(rank);
    Timing::free_timers();

    MPI_Finalize();

    return 0;
}