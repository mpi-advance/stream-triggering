#include "../common/queue_setup.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    int mode;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Info hint for buffer
    MPI_Info mem_info = Common::make_gpu_mem_info();

    // Make HIP variables
    int BUFFER_SIZE = 10;
    int BLOCK_SIZE  = 128;
    int NUM_BLOCKS  = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE * 2, mem_info, &send_buf);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE * 2, mem_info, &recv_buf);

    init_buffers2<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buf, (int*)recv_buf, BUFFER_SIZE,
                                              rank);

    device_sync();
    StreamType my_stream = Setup::create_stream();

    // Make queue
    MPIS_Queue my_queue;
    Setup::create_queue(&my_queue, &my_stream);

    // Make requests
    MPIS_Request my_reqs[2];
    MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD,
                   mem_info, &my_reqs[0]);
    MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, (rank - 1 + size) % size, 0,
                   MPI_COMM_WORLD, mem_info, &my_reqs[1]);

    MPIS_Matchall(2, my_reqs, MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        MPIS_Enqueue_startall(my_queue, 2, my_reqs);
        MPIS_Enqueue_waitall(my_queue, 2, my_reqs, MPI_STATUSES_IGNORE);
    }
    else
    {
        MPIS_Enqueue_start(my_queue, &my_reqs[1]);
        MPIS_Enqueue_wait(my_queue, &my_reqs[1], MPI_STATUS_IGNORE);
        pack_buffer3<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
            (int*)send_buf, (int*)recv_buf, BUFFER_SIZE);
        MPIS_Enqueue_start(my_queue, &my_reqs[0]);
        MPIS_Enqueue_wait(my_queue, &my_reqs[0], MPI_STATUS_IGNORE);
    }

    MPIS_Queue_wait(&my_queue);

    device_sync();
    print_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)recv_buf, BUFFER_SIZE,
                                                            rank);
    device_sync();

    // Cleanup
    MPIS_Request_freeall(2, my_reqs);
    MPIS_Free_mem(send_buf);
    MPIS_Free_mem(recv_buf);
    MPI_Info_free(&mem_info);
    MPIS_Queue_free(&my_queue);

    std::cout << rank << " is done." << std::endl;

    MPI_Finalize();

    return 0;
}
