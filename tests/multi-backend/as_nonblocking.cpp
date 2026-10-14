// Test that creates then frees many objects (basically as if it was only non-blocking and
// not persistant)
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

    int iterations = Checks::iterations(500);

    // The even rank of each pair's data
    int expected_base = 100 * (rank & ~1);

    for (int i = 0; i < iterations; ++i)
    {
        if (Checks::verbose)
            std::cout << rank << " starting iteration: " << i << std::endl;
        // Make requests
        MPIS_Request my_reqs[2];
        if (0 == rank % 2)
        {
            MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, rank + 1, 0, MPI_COMM_WORLD,
                           mem_info, &my_reqs[0]);
            MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, rank + 1, 0, MPI_COMM_WORLD,
                           mem_info, &my_reqs[1]);
        }
        else
        {
            MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, rank - 1, 0, MPI_COMM_WORLD,
                           mem_info, &my_reqs[1]);
            MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, rank - 1, 0, MPI_COMM_WORLD,
                           mem_info, &my_reqs[0]);
        }
        MPIS_Matchall(2, my_reqs, MPI_STATUSES_IGNORE);

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
            Checks::expect_buffer(my_stream, recv_buf, BUFFER_SIZE, expected_base, 1);

            pack_buffer2<<<NUM_BLOCKS, Common::BLOCK_SIZE, 0, my_stream>>>(
                (int*)send_buf, (int*)recv_buf, BUFFER_SIZE);

#ifdef THREAD_BACKEND
            device_sync();
#endif
            MPIS_Enqueue_start(my_queue, &my_reqs[0]);
            MPIS_Enqueue_wait(my_queue, &my_reqs[0], MPI_STATUS_IGNORE);
        }

        MPIS_Queue_wait(&my_queue);
        MPIS_Request_freeall(2, my_reqs);
    }

    // Final check: the odd rank echoed the even rank's data back
    Checks::expect_buffer(my_stream, recv_buf, BUFFER_SIZE, expected_base, 1);

    return Checks::teardown("as_nonblocking");
}