#include "../common/checks.hpp"

/* MPIS_Queue_test reports work as unfinished while a receive is still waiting for its
 * message, and as finished after MPIS_Queue_wait. */
int main(int argc, char* argv[])
{
    using namespace Checks;

    const int BUFFER_SIZE = Checks::buffer_size(64);

    Checks::setup(&argc, &argv, BUFFER_SIZE);
    int  peer     = rank ^ 1;
    bool receiver = (1 == rank % 2);

    MPIS_Request my_req;
    if (receiver)
    {
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                       &my_req);
    }
    else
    {
        MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                       &my_req);
    }
    Common::match_in_place(1, &my_req);

    int flag = -1;
    MPIS_Queue_test(&my_queue, &flag);
    EXPECT(1 == flag, "empty queue reported unfinished work");

    if (receiver)
    {
        MPIS_Enqueue_start(my_queue, &my_req);
        MPIS_Enqueue_wait(my_queue, &my_req, MPI_STATUS_IGNORE);
        MPIS_Queue_test(&my_queue, &flag);
        EXPECT(0 == flag, "queue reported finished before the message was sent");
        /* Only now let the sender go */
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPIS_Enqueue_start(my_queue, &my_req);
        MPIS_Enqueue_wait(my_queue, &my_req, MPI_STATUS_IGNORE);
    }

    MPIS_Queue_wait(&my_queue);
    MPIS_Queue_test(&my_queue, &flag);
    EXPECT(1 == flag, "queue reported unfinished work after MPIS_Queue_wait");

    MPIS_Request_free(&my_req);
    return Checks::teardown("queue_test");
}
