#include "../common/checks.hpp"

/* The bindings reject requests used in the wrong lifecycle state: starting a request
 * that isn't matched (or whose match hasn't completed), freeing a request whose match
 * hasn't completed, matching a matched request, and starting a request again before a
 * wait was enqueued for it. A rejected call changes nothing. */
int main(int argc, char* argv[])
{
    using namespace Checks;

    const int BUFFER_SIZE = Checks::buffer_size(64);

    Checks::setup(&argc, &argv, BUFFER_SIZE);
    int peer = rank ^ 1;

    /* [0] = send, [1] = recv */
    MPIS_Request init_reqs[2];
    MPIS_Request my_reqs[2];
    MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &init_reqs[0]);
    MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &init_reqs[1]);

    /* The originals can't be started */
    EXPECT(Checks::throws([&] { MPIS_Enqueue_start(my_queue, &init_reqs[1]); }),
           "starting an unmatched request was accepted");

    /* Match started but not completed */
    MPIS_Request match_req;
    MPIS_Imatchall(2, init_reqs, my_reqs, MPI_INFO_NULL, &match_req);
    EXPECT(Checks::throws([&] { MPIS_Enqueue_start(my_queue, &my_reqs[1]); }),
           "starting a request whose match hasn't completed was accepted");
    EXPECT(Checks::throws([&] { MPIS_Enqueue_start(my_queue, &match_req); }),
           "starting a match request was accepted");
    EXPECT(Checks::throws([&] { MPIS_Request_free(&my_reqs[1]); }),
           "freeing a request whose match hasn't completed was accepted");
    MPIS_Wait(&match_req, MPI_STATUS_IGNORE);

    /* Waiting on a handle that is now MPIS_REQUEST_NULL is a no-op */
    EXPECT(!Checks::throws([&] { MPIS_Wait(&match_req, MPI_STATUS_IGNORE); }),
           "waiting on a null match request was rejected");

    /* Matching leaves the originals unmatched, and a matched request can't be
     * matched again */
    EXPECT(Checks::throws([&] { MPIS_Enqueue_start(my_queue, &init_reqs[1]); }),
           "starting an original after matching it was accepted");
    EXPECT(Checks::throws([&] {
               MPIS_Request rematched;
               MPIS_Match(my_reqs[1], &rematched, MPI_INFO_NULL);
           }),
           "matching an already matched request was accepted");
    MPIS_Request_freeall(2, init_reqs);

    /* Started, and started again before any wait */
    MPIS_Enqueue_startall(my_queue, 2, my_reqs);
    EXPECT(Checks::throws([&] { MPIS_Enqueue_start(my_queue, &my_reqs[1]); }),
           "restarting a request before waiting was accepted");

    /* The rejected calls changed nothing: the original starts complete normally */
    MPIS_Enqueue_waitall(my_queue, 2, my_reqs, MPI_STATUSES_IGNORE);
    MPIS_Queue_wait(&my_queue);

    /* After the wait is enqueued, starting again is fine */
    EXPECT(!Checks::throws([&] {
        MPIS_Enqueue_startall(my_queue, 2, my_reqs);
        MPIS_Enqueue_waitall(my_queue, 2, my_reqs, MPI_STATUSES_IGNORE);
    }),
           "restarting after a wait was rejected");
    MPIS_Queue_wait(&my_queue);

    MPIS_Request_freeall(2, my_reqs);

    return Checks::teardown("invalid_request_state");
}
