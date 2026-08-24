#define USE_STREAM_TRIGGERING  // For the header to compile with stream triggering stuff
#include "benchmark.hpp"

int main(int argc, char* argv[])
{
    // Includes creating queue.
    Benchmark::init_benchmark<true, true>(&argc, &argv);

#define SEND_REQ (Benchmark::rank ^ 1)
#define RECV_REQ (Benchmark::rank & 1)

    // Make requests
    MPIS_Request my_reqs[2];
    MPIS_Request my_other_reqs[2];
    int          offset = sizeof(int) * Benchmark::BUFFER_SIZE;
    if (0 == Benchmark::rank)
    {
        MPIS_Rsend_init(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                        MPI_COMM_WORLD, Benchmark::mem_info, &my_reqs[SEND_REQ]);
        MPIS_Recv_init(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                       MPI_COMM_WORLD, Benchmark::mem_info, &my_reqs[RECV_REQ]);
        MPIS_Rsend_init((char*)Benchmark::send_buffer + offset, Benchmark::BUFFER_SIZE,
                        MPI_INT, 1, 0, MPI_COMM_WORLD, Benchmark::mem_info,
                        &my_other_reqs[SEND_REQ]);
        MPIS_Recv_init((char*)Benchmark::recv_buffer + offset, Benchmark::BUFFER_SIZE,
                       MPI_INT, 1, 0, MPI_COMM_WORLD, Benchmark::mem_info,
                       &my_other_reqs[RECV_REQ]);
    }
    else
    {
        MPIS_Recv_init(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                       MPI_COMM_WORLD, Benchmark::mem_info, &my_reqs[RECV_REQ]);
        MPIS_Rsend_init(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                        MPI_COMM_WORLD, Benchmark::mem_info, &my_reqs[SEND_REQ]);
        MPIS_Recv_init((char*)Benchmark::recv_buffer + offset, Benchmark::BUFFER_SIZE,
                       MPI_INT, 0, 0, MPI_COMM_WORLD, Benchmark::mem_info,
                       &my_other_reqs[RECV_REQ]);
        MPIS_Rsend_init((char*)Benchmark::send_buffer + offset, Benchmark::BUFFER_SIZE,
                        MPI_INT, 0, 0, MPI_COMM_WORLD, Benchmark::mem_info,
                        &my_other_reqs[SEND_REQ]);
    }

    MPIS_Match(&my_reqs[0], MPI_STATUS_IGNORE);
    MPIS_Match(&my_reqs[1], MPI_STATUS_IGNORE);
    MPIS_Match(&my_other_reqs[0], MPI_STATUS_IGNORE);
    MPIS_Match(&my_other_reqs[1], MPI_STATUS_IGNORE);

    auto do_cycles = [&](int num_cycles) {
        using namespace Benchmark;
        void* active_send_buffer = send_buffer;
        void* active_recv_buffer = recv_buffer;

        void* inactive_send_buffer = (char*)send_buffer + offset;
        void* inactive_recv_buffer = (char*)recv_buffer + offset;

        MPIS_Request* active_request_ptr   = my_reqs;
        MPIS_Request* inactive_request_ptr = my_other_reqs;
        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == rank)
            {
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(my_queue);
#endif
                // Ping side
                pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)active_send_buffer, BUFFER_SIZE, i);
#ifdef THREAD_BACKEND
                device_sync();
#endif
                MPIS_Enqueue_startall(my_queue, 2, active_request_ptr);
                MPIS_Enqueue_waitall(my_queue);

                // #ifdef THREAD_BACKEND
                //                 MPIS_Queue_wait(my_queue);
                // #endif
                //                 print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0,
                //                 bench_stream>>>(
                //                     (int*)active_recv_buffer, BUFFER_SIZE, i, rank);
            }
            else
            {
                MPIS_Enqueue_start(my_queue, &active_request_ptr[RECV_REQ]);
                MPIS_Enqueue_waitall(my_queue);
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(my_queue);
#endif
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                //     (int*)active_recv_buffer, BUFFER_SIZE, i, rank);
                pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)active_send_buffer, (int*)active_recv_buffer, BUFFER_SIZE);
#ifdef THREAD_BACKEND
                device_sync();
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

    Benchmark::run_experiment(do_cycles, true);

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    MPIS_Request_freeall(2, my_other_reqs);
    Benchmark::cleanup();
    MPI_Finalize();

    return 0;
}