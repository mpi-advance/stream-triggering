#include "../common/checks.hpp"

/* MPIS_Enqueue_startall and MPIS_Enqueue_waitall reject a list that names the same
 * request more than once, without changing anything. */
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
    MPIS_Free_mem(send_buf);
    MPIS_Free_mem(recv_buf);
    MPI_Info_free(&mem_info);
    MPIS_Queue_free(&my_queue);

    return Checks::finish("duplicate_requests");
}
