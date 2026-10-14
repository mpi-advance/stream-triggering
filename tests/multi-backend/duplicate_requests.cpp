#include "../common/checks.hpp"

/* MPIS_Enqueue_startall and MPIS_Enqueue_waitall reject a list that names the same
 * request more than once, without changing anything. */
int main(int argc, char* argv[])
{
    using namespace Checks;

    const int BUFFER_SIZE = Checks::buffer_size(64);

    Checks::setup(&argc, &argv, BUFFER_SIZE);
    int peer = rank ^ 1;

    /* [0] = send, [1] = recv */
    MPIS_Request my_reqs[2];
    MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &my_reqs[0]);
    MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &my_reqs[1]);
    MPIS_Matchall(2, my_reqs, MPI_STATUSES_IGNORE);

    MPIS_Request twice[3] = {my_reqs[0], my_reqs[1], my_reqs[1]};
    EXPECT(Checks::throws([&] { MPIS_Enqueue_startall(my_queue, 3, twice); }),
           "startall naming a request twice was accepted");

    /* Nothing was started, so a normal startall still works */
    MPIS_Enqueue_startall(my_queue, 2, my_reqs);
    EXPECT(Checks::throws(
               [&] { MPIS_Enqueue_waitall(my_queue, 3, twice, MPI_STATUSES_IGNORE); }),
           "waitall naming a request twice was accepted");

    /* Nothing was waited on either, so a normal waitall completes the starts */
    MPIS_Enqueue_waitall(my_queue, 2, my_reqs, MPI_STATUSES_IGNORE);
    MPIS_Queue_wait(&my_queue);

    MPIS_Request_freeall(2, my_reqs);

    return Checks::teardown("duplicate_requests");
}
