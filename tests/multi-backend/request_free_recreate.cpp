#include "../common/checks.hpp"

/* Freeing persistent requests mid-run and creating new ones in their place works: the
 * new requests communicate normally on the same queue. */
int main(int argc, char* argv[])
{
    using namespace Checks;

    const int BUFFER_SIZE = Checks::buffer_size(64);
    const int ROUNDS      = 5;

    Checks::setup(&argc, &argv, BUFFER_SIZE);
    int peer = rank ^ 1;

    for (int round = 0; round < ROUNDS; ++round)
    {
        /* [0] = send, [1] = recv; a different tag each round */
        MPIS_Request my_reqs[2];
        MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, round, MPI_COMM_WORLD,
                       mem_info, &my_reqs[0]);
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, round, MPI_COMM_WORLD,
                       mem_info, &my_reqs[1]);
        MPIS_Matchall(2, my_reqs, MPI_STATUSES_IGNORE);

        MPI_Status statuses[2];
        MPIS_Enqueue_startall(my_queue, 2, my_reqs);
        MPIS_Enqueue_waitall(my_queue, 2, my_reqs, statuses);
        MPIS_Queue_wait(&my_queue);

        if (P2P_REPORTS_STATUS)
        {
            EXPECT(peer == statuses[1].MPI_SOURCE && round == statuses[1].MPI_TAG,
                   "round %d: recv status source %d tag %d", round,
                   statuses[1].MPI_SOURCE, statuses[1].MPI_TAG);
        }

        MPIS_Request_freeall(2, my_reqs);
    }

    return Checks::teardown("request_free_recreate");
}
