#include "../common/checks.hpp"

/* MPIS_Enqueue_wait and MPIS_Enqueue_waitall fill in real MPI_Statuses for the requests
 * they wait on, every time the same persistent requests are reused. */
int main(int argc, char* argv[])
{
    using namespace Checks;

    const int BUFFER_SIZE = Checks::buffer_size(64);
    const int ITERATIONS  = Checks::iterations(20);

    Checks::setup(&argc, &argv, BUFFER_SIZE);
    int peer = rank ^ 1;

    /* [0] = send, [1] = recv */
    MPIS_Request my_reqs[2];
    MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &my_reqs[0]);
    MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &my_reqs[1]);
    Common::match_in_place(2, my_reqs);

    for (int i = 0; i < ITERATIONS; ++i)
    {
        MPI_Status recv_status;
        MPIS_Enqueue_startall(my_queue, 2, my_reqs);
        if (0 == i % 2)
        {
            MPI_Status statuses[2];
            MPIS_Enqueue_waitall(my_queue, 2, my_reqs, statuses);
            MPIS_Queue_wait(&my_queue);
            recv_status = statuses[1];
        }
        else
        {
            MPIS_Enqueue_wait(my_queue, &my_reqs[1], &recv_status);
            MPIS_Enqueue_wait(my_queue, &my_reqs[0], MPI_STATUS_IGNORE);
            MPIS_Queue_wait(&my_queue);
        }

        if (P2P_REPORTS_STATUS)
        {
            int count = -1;
            MPI_Get_count(&recv_status, MPI_INT, &count);
            EXPECT(peer == recv_status.MPI_SOURCE && 5 == recv_status.MPI_TAG &&
                       BUFFER_SIZE == count,
                   "iteration %d: recv status source %d tag %d count %d", i,
                   recv_status.MPI_SOURCE, recv_status.MPI_TAG, count);
        }
        else
        {
            EXPECT(Checks::is_empty_status(recv_status),
                   "iteration %d: expected an empty status", i);
        }
    }

    MPIS_Request_freeall(2, my_reqs);

    return Checks::teardown("enqueue_wait_statuses");
}
