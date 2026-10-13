#include "../common/queue_setup.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    int mode;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Input parameters
    int BUFFER_SIZE = 64;

    // Info hint for buffer
    MPI_Info mem_info = Common::make_gpu_mem_info();

    // Make Buffers
    int BLOCK_SIZE = 128;
    int NUM_BLOCKS = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &send_buf);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &recv_buf);

    init_buffers2<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buf, (int*)recv_buf, BUFFER_SIZE,
                                              rank);
    device_sync();

    StreamType my_stream = Setup::create_stream();

    // Make queue
    MPIS_Queue my_queue;
    Setup::create_queue(&my_queue, &my_stream);

    // Make requests
    MPIS_Request my_reqs[2];
    if (0 == rank % 2)
    {
        MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[0]);
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[1]);
    }
    else
    {
        MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[1]);
        MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, mem_info,
                       &my_reqs[0]);
    }
    MPIS_Matchall(2, my_reqs, MPI_STATUSES_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << rank << " starting test!" << std::endl;

    double start = MPI_Wtime();
    if (0 == rank)
    {
        MPIS_Enqueue_startall(my_queue, 2, my_reqs);
        MPIS_Enqueue_waitall(my_queue, 2, my_reqs, MPI_STATUSES_IGNORE);
    }
    else
    {
        MPIS_Enqueue_start(my_queue, &my_reqs[1]);
        MPIS_Enqueue_wait(my_queue, &my_reqs[1], MPI_STATUS_IGNORE);

#ifdef THREAD_BACKEND
        MPIS_Queue_wait(&my_queue);
#endif
        print_buffer3<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)recv_buf,
                                                                BUFFER_SIZE, rank);

        pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
            (int*)send_buf, (int*)recv_buf, BUFFER_SIZE);

#ifdef THREAD_BACKEND
        device_sync();
#endif
        MPIS_Enqueue_start(my_queue, &my_reqs[0]);
        MPIS_Enqueue_wait(my_queue, &my_reqs[0], MPI_STATUS_IGNORE);
    }

    MPIS_Queue_wait(&my_queue);
    double end = MPI_Wtime();

    std::cout << rank << " done with GPU communication" << std::endl;
    std::cout << rank << " time: " << end - start << std::endl;

    // Final check
    device_sync();
    dummy_kernel<<<1, 1, 0, my_stream>>>(rank, 2);
    print_buffer3<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)recv_buf, BUFFER_SIZE,
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