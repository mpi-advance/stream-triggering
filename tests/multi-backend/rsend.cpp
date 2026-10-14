#include "../common/checks.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    using namespace Checks;

    // Input parameters
    int BUFFER_SIZE = Checks::buffer_size(64);

    // Buffers (send: 100 * rank + index), stream and queue
    Checks::setup(&argc, &argv, 2 * BUFFER_SIZE);

    // Make requests
    int          peer = rank ^ 1;
    MPIS_Request my_reqs[2];
    int          offset = sizeof(int) * BUFFER_SIZE;
    if (0 == rank % 2)
    {
        MPIS_Rsend_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD, mem_info,
                        &my_reqs[0]);
        MPIS_Send_init((char*)send_buf + offset, BUFFER_SIZE, MPI_INT, peer, 0,
                       MPI_COMM_WORLD, mem_info, &my_reqs[1]);
    }
    else
    {
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[0]);
        MPIS_Recv_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, peer, 0,
                       MPI_COMM_WORLD, mem_info, &my_reqs[1]);
    }

    Common::match_in_place(2, my_reqs);
    MPI_Barrier(MPI_COMM_WORLD);

    if (Checks::verbose)
        std::cout << rank << " starting test!" << std::endl;
    double start = MPI_Wtime();
    if (1 == rank % 2)
    {
        sleep(5);
    }
    MPIS_Enqueue_start(my_queue, &my_reqs[0]);
    MPIS_Enqueue_wait(my_queue, &my_reqs[0], MPI_STATUS_IGNORE);
    MPIS_Queue_wait(&my_queue);
    double end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    if (Checks::verbose)
        std::cout << rank << " time A: " << end - start << std::endl;

    start = MPI_Wtime();
    if (1 == rank % 2)
    {
        sleep(5);
    }
    MPIS_Enqueue_start(my_queue, &my_reqs[1]);
    MPIS_Enqueue_wait(my_queue, &my_reqs[1], MPI_STATUS_IGNORE);

    MPIS_Queue_wait(&my_queue);
    end = MPI_Wtime();

    if (Checks::verbose)
        std::cout << rank << " time B: " << end - start << std::endl;

    // Final check: the even rank's data, both halves
    if (1 == rank % 2)
    {
        Checks::expect_buffer(my_stream, recv_buf, BUFFER_SIZE * 2, 100 * (rank & ~1), 1);
    }

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    return Checks::teardown("rsend");
}