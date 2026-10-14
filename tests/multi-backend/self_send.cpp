#include "../common/checks.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    using namespace Checks;

    // Input parameters
    int BUFFER_SIZE = Checks::buffer_size(64);

    // Buffers (send: 100 * rank + index), stream and queue
    Checks::setup(&argc, &argv, BUFFER_SIZE, 1);

    // Make requests
    MPIS_Request my_reqs[2];
    MPIS_Rsend_init(send_buf, BUFFER_SIZE, MPI_INT, rank, 0, MPI_COMM_WORLD, mem_info,
                    &my_reqs[0]);
    MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, rank, 0, MPI_COMM_WORLD, mem_info,
                   &my_reqs[1]);

    MPIS_Matchall(2, my_reqs, MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    if (Checks::verbose)
        std::cout << rank << " starting test!" << std::endl;

    double start = MPI_Wtime();
    MPIS_Enqueue_startall(my_queue, 2, my_reqs);
    MPIS_Enqueue_waitall(my_queue, 2, my_reqs, MPI_STATUSES_IGNORE);
    MPIS_Queue_wait(&my_queue);
    double end = MPI_Wtime();

    if (Checks::verbose)
    {
        std::cout << rank << " done with GPU communication" << std::endl;
        std::cout << rank << " time: " << end - start << std::endl;
    }

    // Final check: this rank's own data
    Checks::expect_buffer(my_stream, recv_buf, BUFFER_SIZE, 100 * rank, 1);

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    return Checks::teardown("self_send");
}