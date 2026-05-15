#include "benchmark.hpp"

int main(int argc, char* argv[])
{
    Benchmark::init_benchmark<true, true>(&argc, &argv);

#define SEND_REQ (Benchmark::rank ^ 1)
#define RECV_REQ (Benchmark::rank & 1)
    // Make requests
    MPI_Request my_reqs[2];
    MPI_Request my_other_reqs[2];
    int         offset = sizeof(int) * Benchmark::BUFFER_SIZE;
    if (0 == Benchmark::rank)
    {
        MPI_Rsend_init(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                       MPI_COMM_WORLD, &my_reqs[SEND_REQ]);
        MPI_Recv_init(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                      MPI_COMM_WORLD, &my_reqs[RECV_REQ]);
        MPI_Rsend_init((char*)Benchmark::send_buffer + offset, Benchmark::BUFFER_SIZE,
                       MPI_INT, 1, 0, MPI_COMM_WORLD, &my_other_reqs[SEND_REQ]);
        MPI_Recv_init((char*)Benchmark::recv_buffer + offset, Benchmark::BUFFER_SIZE,
                      MPI_INT, 1, 0, MPI_COMM_WORLD, &my_other_reqs[RECV_REQ]);
    }
    else
    {
        MPI_Recv_init(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                      MPI_COMM_WORLD, &my_reqs[RECV_REQ]);
        MPI_Rsend_init(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                       MPI_COMM_WORLD, &my_reqs[SEND_REQ]);
        MPI_Recv_init((char*)Benchmark::recv_buffer + offset, Benchmark::BUFFER_SIZE,
                      MPI_INT, 0, 0, MPI_COMM_WORLD, &my_other_reqs[RECV_REQ]);
        MPI_Rsend_init((char*)Benchmark::send_buffer + offset, Benchmark::BUFFER_SIZE,
                       MPI_INT, 0, 0, MPI_COMM_WORLD, &my_other_reqs[SEND_REQ]);
    }

    auto do_cycles = [&](int num_cycles) {
        using namespace Benchmark;
        void* active_send_buffer = Benchmark::send_buffer;
        void* active_recv_buffer = Benchmark::recv_buffer;

        void* inactive_send_buffer = (char*)Benchmark::send_buffer + offset;
        void* inactive_recv_buffer = (char*)Benchmark::recv_buffer + offset;

        MPI_Request* active_request_ptr   = my_reqs;
        MPI_Request* inactive_request_ptr = my_other_reqs;

        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == rank)
            {
                // Ping side
                pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)active_send_buffer, BUFFER_SIZE, i);

                stream_sync(bench_stream);
                MPI_Startall(2, active_request_ptr);
                MPI_Waitall(2, active_request_ptr, MPI_STATUSES_IGNORE);
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                //     (int*)active_recv_buffer, BUFFER_SIZE, i, rank);
            }
            else
            {
                MPI_Start(&active_request_ptr[RECV_REQ]);
                MPI_Wait(&active_request_ptr[RECV_REQ], MPI_STATUS_IGNORE);
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                //     (int*)active_recv_buffer, BUFFER_SIZE, i, rank);
                pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)active_send_buffer, (int*)active_recv_buffer, BUFFER_SIZE);
                stream_sync(bench_stream);
                MPI_Start(&active_request_ptr[SEND_REQ]);
                MPI_Wait(&active_request_ptr[SEND_REQ], MPI_STATUS_IGNORE);
            }

            void* temp_send      = active_send_buffer;
            active_send_buffer   = inactive_send_buffer;
            inactive_send_buffer = temp_send;

            void* temp_recv      = active_recv_buffer;
            active_recv_buffer   = inactive_recv_buffer;
            inactive_recv_buffer = temp_recv;

            MPI_Request* temp_reqs = active_request_ptr;
            active_request_ptr     = inactive_request_ptr;
            inactive_request_ptr   = temp_reqs;
        }
    };
    Benchmark::run_experiment(do_cycles, true);

    // Cleanup
    MPI_Request_free(&my_reqs[SEND_REQ]);
    MPI_Request_free(&my_reqs[RECV_REQ]);
    MPI_Request_free(&my_other_reqs[SEND_REQ]);
    MPI_Request_free(&my_other_reqs[RECV_REQ]);
    MPI_Finalize();

    Benchmark::cleanup();
    return 0;
}