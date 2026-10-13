#include "../common/checks.hpp"

/* The bindings reject requests used in the wrong lifecycle state: starting a request
 * that isn't matched (or whose match hasn't completed), and starting a request again
 * before a wait was enqueued for it. A rejected call changes nothing. */
int main(int argc, char* argv[])
{
    Checks::init(&argc, &argv);
    int       peer        = Checks::rank ^ 1;
    const int BUFFER_SIZE = 64;

    MPI_Info mem_info = Common::make_gpu_mem_info();
    void*    send_buf = nullptr;
    void*    recv_buf = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &send_buf);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &recv_buf);

    StreamType my_stream = Setup::create_stream();
    MPIS_Queue my_queue;
    Setup::create_queue(&my_queue, &my_stream);

    /* [0] = send, [1] = recv */
    MPIS_Request my_reqs[2];
    MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &my_reqs[0]);
    MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                   &my_reqs[1]);

    /* Not matched yet */
    EXPECT(Checks::throws([&] { MPIS_Enqueue_start(my_queue, &my_reqs[1]); }),
           "starting an unmatched request was accepted");

    /* Match started but not completed */
    MPIS_Request match_handles[2];
    MPIS_Imatch(&my_reqs[0], &match_handles[0]);
    MPIS_Imatch(&my_reqs[1], &match_handles[1]);
    EXPECT(Checks::throws([&] { MPIS_Enqueue_start(my_queue, &my_reqs[1]); }),
           "starting a request whose match hasn't completed was accepted");
    EXPECT(Checks::throws([&] { MPIS_Enqueue_start(my_queue, &match_handles[1]); }),
           "starting a match handle was accepted");
    MPIS_Waitall(2, match_handles, MPI_STATUSES_IGNORE);

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
    MPIS_Free_mem(send_buf);
    MPIS_Free_mem(recv_buf);
    MPI_Info_free(&mem_info);
    MPIS_Queue_free(&my_queue);

    return Checks::finish("invalid_request_state");
}
