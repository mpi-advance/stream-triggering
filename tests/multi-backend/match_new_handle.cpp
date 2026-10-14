#include "../common/checks.hpp"

/* Matching returns new handles and leaves the originals alone: the same originals can
 * be matched more than once, they can be freed before the matched handles are used,
 * and each set of matched handles carries its own traffic. Freeing a match request
 * instead of waiting on it still completes the match. */
int main(int argc, char* argv[])
{
    using namespace Checks;

    const int BUFFER_SIZE = Checks::buffer_size(64);

    Checks::setup(&argc, &argv, BUFFER_SIZE);
    int peer = rank ^ 1;

    /* [0] = send, [1] = recv */
    MPIS_Request init_reqs[2];
    MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &init_reqs[0]);
    MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &init_reqs[1]);

    /* Match the same originals three times: blocking, nonblocking then waited, and
     * nonblocking then freed */
    MPIS_Request first[2];
    MPIS_Request second[2];
    MPIS_Request third[2];
    MPIS_Request match_req;
    MPIS_Matchall(2, init_reqs, first, MPI_INFO_NULL);
    MPIS_Imatchall(2, init_reqs, second, MPI_INFO_NULL, &match_req);
    MPIS_Wait(&match_req, MPI_STATUS_IGNORE);
    EXPECT(MPIS_REQUEST_NULL == match_req, "the match request wasn't reset by the wait");
    MPIS_Imatchall(2, init_reqs, third, MPI_INFO_NULL, &match_req);
    MPIS_Request_free(&match_req);
    EXPECT(MPIS_REQUEST_NULL == match_req, "the match request wasn't reset by the free");

    int matched = -1;
    MPIS_Is_matched(&init_reqs[0], &matched);
    EXPECT(0 == matched, "an original reported itself matched");
    for (int i = 0; i < 2; ++i)
    {
        MPIS_Is_matched(&first[i], &matched);
        EXPECT(1 == matched, "a handle from MPIS_Matchall isn't matched");
        MPIS_Is_matched(&second[i], &matched);
        EXPECT(1 == matched, "a handle from MPIS_Imatchall isn't matched after the wait");
        MPIS_Is_matched(&third[i], &matched);
        EXPECT(1 == matched,
               "a handle from MPIS_Imatchall isn't matched after freeing its match");
    }

    /* The originals aren't needed once matched */
    MPIS_Request_freeall(2, init_reqs);

    for (MPIS_Request* reqs : {first, second, third})
    {
        MPIS_Enqueue_startall(my_queue, 2, reqs);
        MPIS_Enqueue_waitall(my_queue, 2, reqs, MPI_STATUSES_IGNORE);
        MPIS_Queue_wait(&my_queue);
        Checks::expect_buffer(my_stream, recv_buf, BUFFER_SIZE, 100 * peer, 1);

        /* Reset the receive buffer, so the next set has to deliver the data again */
        init_buffers2<<<Common::num_blocks(BUFFER_SIZE), Common::BLOCK_SIZE, 0,
                        my_stream>>>((int*)send_buf, (int*)recv_buf, BUFFER_SIZE, rank);
        device_sync();
    }

    MPIS_Request_freeall(2, first);
    MPIS_Request_freeall(2, second);
    MPIS_Request_freeall(2, third);

    return Checks::teardown("match_new_handle");
}
