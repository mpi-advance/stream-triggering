#define USE_STREAM_TRIGGERING  // For the header to compile with stream triggering stuff
#include "benchmark.hpp"

int main(int argc, char* argv[])
{
    // Includes creating queue.
    Benchmark::init_benchmark<true, false>(&argc, &argv);

#define SEND_REQ (Benchmark::rank ^ 1)
#define RECV_REQ (Benchmark::rank & 1)

    // Make requests
    MPIS_Request my_reqs[2];
    if (0 == Benchmark::rank)
    {
        MPIS_Send_init(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                       MPI_COMM_WORLD, Benchmark::mem_info, &my_reqs[SEND_REQ]);
        MPIS_Recv_init(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                       MPI_COMM_WORLD, Benchmark::mem_info, &my_reqs[RECV_REQ]);
    }
    else
    {
        MPIS_Recv_init(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                       MPI_COMM_WORLD, Benchmark::mem_info, &my_reqs[RECV_REQ]);
        MPIS_Send_init(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                       MPI_COMM_WORLD, Benchmark::mem_info, &my_reqs[SEND_REQ]);
    }

    MPIS_Match(&my_reqs[0], MPI_STATUS_IGNORE);
    MPIS_Match(&my_reqs[1], MPI_STATUS_IGNORE);

    auto do_cycles = [&](int num_cycles) {
        using namespace Benchmark;
        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == rank)
            {
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(my_queue);
#endif
                // Ping side
                pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)send_buffer, BUFFER_SIZE, i);
#ifdef THREAD_BACKEND
                device_sync();
#endif
                MPIS_Enqueue_startall(my_queue, 2, my_reqs);
                MPIS_Enqueue_waitall(my_queue);

                // #ifdef THREAD_BACKEND
                //                 MPIS_Queue_wait(my_queue);
                // #endif
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                //     (int*)recv_buffer, BUFFER_SIZE, i, rank);
            }
            else
            {
                MPIS_Enqueue_start(my_queue, &my_reqs[RECV_REQ]);
                MPIS_Enqueue_waitall(my_queue);
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(my_queue);
#endif
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                //     (int*)recv_buffer, BUFFER_SIZE, i, rank);
                pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)send_buffer, (int*)recv_buffer, BUFFER_SIZE);
#ifdef THREAD_BACKEND
                device_sync();
#endif
                MPIS_Enqueue_start(my_queue, &my_reqs[SEND_REQ]);
                MPIS_Enqueue_waitall(my_queue);
            }
        }

        MPIS_Queue_wait(my_queue);
    };

    Benchmark::run_experiment(do_cycles);

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    Benchmark::cleanup();
    MPI_Finalize();

    return 0;
}