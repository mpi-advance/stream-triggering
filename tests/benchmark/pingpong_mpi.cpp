#include "benchmark.hpp"

int main(int argc, char* argv[])
{
    Benchmark::init_benchmark<true, false>(&argc, &argv);

#define SEND_REQ (Benchmark::rank ^ 1)
#define RECV_REQ (Benchmark::rank & 1)

    // Make requests
    MPI_Request my_reqs[2];
    if (0 == Benchmark::rank)
    {
        MPI_Send_init(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                      MPI_COMM_WORLD, &my_reqs[SEND_REQ]);
        MPI_Recv_init(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                      MPI_COMM_WORLD, &my_reqs[RECV_REQ]);
    }
    else
    {
        MPI_Recv_init(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                      MPI_COMM_WORLD, &my_reqs[RECV_REQ]);
        MPI_Send_init(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                      MPI_COMM_WORLD, &my_reqs[SEND_REQ]);
    }

    auto do_cycles = [&](int num_cycles) {
        using namespace Benchmark;
        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == rank)
            {
                // Ping side
                pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)send_buffer, BUFFER_SIZE, i);

                stream_sync(bench_stream);
                MPI_Startall(2, my_reqs);
                MPI_Waitall(2, my_reqs, MPI_STATUSES_IGNORE);
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                //    (int*)recv_buffer, BUFFER_SIZE, i, rank);
            }
            else
            {
                MPI_Start(&my_reqs[RECV_REQ]);
                MPI_Wait(&my_reqs[RECV_REQ], MPI_STATUS_IGNORE);
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                //    (int*)recv_buffer, BUFFER_SIZE, i, rank);
                pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)send_buffer, (int*)recv_buffer, BUFFER_SIZE);
                stream_sync(bench_stream);
                MPI_Start(&my_reqs[SEND_REQ]);
                MPI_Wait(&my_reqs[SEND_REQ], MPI_STATUS_IGNORE);
            }
        }
    };

    Benchmark::run_experiment(do_cycles);

    // Cleanup
    MPI_Request_free(&my_reqs[SEND_REQ]);
    MPI_Request_free(&my_reqs[RECV_REQ]);
    MPI_Finalize();

    Benchmark::cleanup();
    return 0;
}