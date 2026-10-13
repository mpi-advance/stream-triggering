#include "../common/checks.hpp"

/* MPIS_Queue_test reports work as unfinished while a receive is still waiting for its
 * message, and as finished after MPIS_Queue_wait. */
int main(int argc, char* argv[])
{
    Checks::init(&argc, &argv);
    int       peer        = Checks::rank ^ 1;
    bool      receiver    = (1 == Checks::rank % 2);
    const int BUFFER_SIZE = 64;

    MPI_Info mem_info = Common::make_gpu_mem_info();
    void*    buffer   = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &buffer);

    StreamType my_stream = Setup::create_stream();
    MPIS_Queue my_queue;
    Setup::create_queue(&my_queue, &my_stream);

    MPIS_Request my_req;
    if (receiver)
    {
        MPIS_Recv_init(buffer, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                       &my_req);
    }
    else
    {
        MPIS_Send_init(buffer, BUFFER_SIZE, MPI_INT, peer, 5, MPI_COMM_WORLD, mem_info,
                       &my_req);
    }
    MPIS_Match(&my_req, MPI_STATUS_IGNORE);

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
    MPIS_Free_mem(buffer);
    MPI_Info_free(&mem_info);
    MPIS_Queue_free(&my_queue);

    return Checks::finish("queue_test");
}
