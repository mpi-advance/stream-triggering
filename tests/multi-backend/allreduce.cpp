#include "../common/queue_setup.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    int mode;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Make request
    MPIS_Request my_req;
    MPIS_Allreduce_init(send_buf, recv_buf, BUFFER_SIZE, MPI_INT, MPI_SUM, MPI_COMM_WORLD, MPI_INFO_NULL, &my_req);
    MPIS_Queue_match(my_queue, &my_req, MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << rank << " starting test!" << std::endl;

    double start = MPI_Wtime();
    MPIS_Enqueue_start(my_queue, &my_req);
    MPIS_Enqueue_wait(my_queue, &my_req, MPI_STATUS_IGNORE);
    MPIS_Queue_wait(&my_queue);
    double end = MPI_Wtime();

    std::cout << rank << " done with GPU communication" << std::endl;
    std::cout << rank << " time: " << end - start << std::endl;

    // Final check
    device_sync();
    verify<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)recv_buf, BUFFER_SIZE,
                                                            rank, size);
    device_sync();

    // Cleanup
    MPIS_Request_freeall(1, &my_req);
    MPIS_Free_mem(send_buf);
    MPIS_Free_mem(recv_buf);

    MPI_Info_free(&mem_info);

    MPIS_Queue_free(&my_queue);

    std::cout << rank << " is done." << std::endl;

    MPI_Finalize();

    return 0;
}