#include "../common/checks.hpp"

/* Requests named in a wait that have no start to wait on (never started, or already
 * waited on) get empty MPI_Statuses, in the caller's order, next to real ones. */
int main(int argc, char* argv[])
{
    using namespace Checks;

    const int BUFFER_SIZE = Checks::buffer_size(64);

    Checks::setup(&argc, &argv, BUFFER_SIZE);
    int   peer     = rank ^ 1;
    void* idle_buf = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE * 2, mem_info, &idle_buf);

    /* [0] = send, [1] = recv, [2]/[3] = a matched pair that is never started */
    MPIS_Request my_reqs[4];
    MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &my_reqs[0]);
    MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &my_reqs[1]);
    MPIS_Send_init(idle_buf, BUFFER_SIZE, MPI_INT, peer, 9, MPI_COMM_WORLD, mem_info,
                   &my_reqs[2]);
    MPIS_Recv_init((int*)idle_buf + BUFFER_SIZE, BUFFER_SIZE, MPI_INT, peer, 9,
                   MPI_COMM_WORLD, mem_info, &my_reqs[3]);
    MPIS_Matchall(4, my_reqs, MPI_STATUSES_IGNORE);

    /* Started recv, never-started recv, and a send whose wait was already enqueued */
    MPIS_Enqueue_startall(my_queue, 2, my_reqs);
    MPIS_Enqueue_wait(my_queue, &my_reqs[0], MPI_STATUS_IGNORE);
    {
        MPIS_Request mixed[3] = {my_reqs[1], my_reqs[3], my_reqs[0]};
        MPI_Status   statuses[3];
        MPIS_Enqueue_waitall(my_queue, 3, mixed, statuses);
        MPIS_Queue_wait(&my_queue);

        if (P2P_REPORTS_STATUS)
        {
            EXPECT(peer == statuses[0].MPI_SOURCE && 5 == statuses[0].MPI_TAG,
                   "started recv: source %d tag %d", statuses[0].MPI_SOURCE,
                   statuses[0].MPI_TAG);
        }
        EXPECT(Checks::is_empty_status(statuses[1]), "never-started recv not empty");
        EXPECT(Checks::is_empty_status(statuses[2]), "already-waited send not empty");
    }

    /* Nothing named has a start to wait on */
    {
        MPI_Status statuses[2];
        MPIS_Enqueue_waitall(my_queue, 2, &my_reqs[2], statuses);
        MPIS_Queue_wait(&my_queue);
        EXPECT(
            Checks::is_empty_status(statuses[0]) && Checks::is_empty_status(statuses[1]),
            "wait with nothing started: statuses not empty");
    }

    MPIS_Request_freeall(4, my_reqs);
    MPIS_Free_mem(idle_buf);

    return Checks::teardown("enqueue_waitall_empty_statuses");
}
