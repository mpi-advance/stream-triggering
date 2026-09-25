#include "benchmark.hpp"

int main(int argc, char* argv[])
{
    Benchmark::init_benchmark<true, false>(&argc, &argv);

    auto do_cycles = [&](int num_cycles) {
        using namespace Benchmark;
        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == Benchmark::rank)
            {
                MPI_Send(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                         MPI_COMM_WORLD);
                MPI_Recv(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                MPI_Recv(Benchmark::recv_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(Benchmark::send_buffer, Benchmark::BUFFER_SIZE, MPI_INT, 0, 0,
                         MPI_COMM_WORLD);
            }
        }
        device_sync();
    };

    Benchmark::run_experiment(do_cycles, false, false);

    // Cleanup
    MPI_Finalize();

    Benchmark::cleanup();
    return 0;
}