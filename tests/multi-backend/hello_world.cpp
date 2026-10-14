#include "../common/checks.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    using namespace Checks;

    // Input parameters
    int BUFFER_SIZE = Checks::buffer_size(64);
    int NUM_BLOCKS  = Common::num_blocks(BUFFER_SIZE);

    // Buffers (send: 100 * rank + index), stream and queue
    Checks::setup(&argc, &argv, BUFFER_SIZE);

    // Make requests
    int          peer = rank ^ 1;
    MPIS_Request my_reqs[2];
    if (0 == rank % 2)
    {
        MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[0]);
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[1]);
    }
    else
    {
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[1]);
        MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[0]);
    }
    MPIS_Matchall(2, my_reqs, MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    if (Checks::verbose)
        std::cout << rank << " starting test!" << std::endl;

    double start = MPI_Wtime();
    if (0 == rank % 2)
    {
        MPIS_Enqueue_startall(my_queue, 2, my_reqs);
        MPIS_Enqueue_waitall(my_queue, 2, my_reqs, MPI_STATUSES_IGNORE);
    }
    else
    {
        MPIS_Enqueue_start(my_queue, &my_reqs[1]);
        MPIS_Enqueue_wait(my_queue, &my_reqs[1], MPI_STATUS_IGNORE);

#ifdef THREAD_BACKEND
        MPIS_Queue_wait(&my_queue);
#endif
        // The even rank's data
        Checks::expect_buffer(my_stream, recv_buf, BUFFER_SIZE, 100 * (rank & ~1), 1);

        pack_buffer2<<<NUM_BLOCKS, Common::BLOCK_SIZE, 0, my_stream>>>(
            (int*)send_buf, (int*)recv_buf, BUFFER_SIZE);

#ifdef THREAD_BACKEND
        device_sync();
#endif
        MPIS_Enqueue_start(my_queue, &my_reqs[0]);
        MPIS_Enqueue_wait(my_queue, &my_reqs[0], MPI_STATUS_IGNORE);
    }

    MPIS_Queue_wait(&my_queue);
    double end = MPI_Wtime();

    if (Checks::verbose)
    {
        std::cout << rank << " done with GPU communication" << std::endl;
        std::cout << rank << " time: " << end - start << std::endl;
    }

    // Final check: the odd rank echoed the even rank's data back
    Checks::expect_buffer(my_stream, recv_buf, BUFFER_SIZE, 100 * (rank & ~1), 1);

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    return Checks::teardown("hello_world");
}
