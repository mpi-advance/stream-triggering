#include "../common/checks.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    using namespace Checks;

    // Input parameters
    int BUFFER_SIZE = Checks::buffer_size(10);
    int NUM_BLOCKS  = Common::num_blocks(BUFFER_SIZE);

    // Buffers (send: 100 * rank + index), stream and queue
    Checks::setup(&argc, &argv, BUFFER_SIZE, 1);

    // Make requests
    MPIS_Request my_reqs[2];
    MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD,
                   mem_info, &my_reqs[0]);
    MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, (rank - 1 + size) % size, 0,
                   MPI_COMM_WORLD, mem_info, &my_reqs[1]);

    Common::match_in_place(2, my_reqs);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
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
        pack_buffer3<<<NUM_BLOCKS, Common::BLOCK_SIZE, 0, my_stream>>>(
            (int*)send_buf, (int*)recv_buf, BUFFER_SIZE);
#ifdef THREAD_BACKEND
        device_sync();
#endif
        MPIS_Enqueue_start(my_queue, &my_reqs[0]);
        MPIS_Enqueue_wait(my_queue, &my_reqs[0], MPI_STATUS_IGNORE);
    }

    MPIS_Queue_wait(&my_queue);

    // Final check: rank r receives the sum of (100 * k + index) over ranks k < r, and
    // rank 0 receives it over all ranks
    int ranks_summed = (0 == rank) ? size : rank;
    Checks::expect_buffer(my_stream, recv_buf, BUFFER_SIZE,
                          100 * (ranks_summed * (ranks_summed - 1) / 2), ranks_summed);

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    return Checks::teardown("ring");
}
