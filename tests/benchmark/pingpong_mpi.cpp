#include "common.hpp"

int main(int argc, char* argv[])
{
    int mode;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // I want "two params"
    check_param_size(&argc, 2,
                     "<program> <number of iterations> <buffer size>");

    // Input parameters
    int num_iters   = 0;
    int BUFFER_SIZE = 0;
    read_iter_buffer_input(&argv, &num_iters, &BUFFER_SIZE);

    // Make Buffers
    int BLOCK_SIZE = 128;
    int NUM_BLOCKS = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    force_hip(hipExtMallocWithFlags(&send_buf, sizeof(int) * BUFFER_SIZE,
                                    hipDeviceMallocFinegrained));
    force_hip(hipExtMallocWithFlags(&recv_buf, sizeof(int) * BUFFER_SIZE,
                                    hipDeviceMallocFinegrained));

    init_buffers<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buf, (int*)recv_buf,
                                             BUFFER_SIZE);
    check_hip(hipDeviceSynchronize());

    hipStream_t my_stream;
    check_hip(hipStreamCreateWithFlags(&my_stream, hipStreamNonBlocking));

#define SEND_REQ (rank ^ 1)
#define RECV_REQ (rank & 1)

    // Make requests
    MPI_Request my_reqs[2];
    if (0 == rank % 2)
    {
        MPI_Send_init(send_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD,
                      &my_reqs[SEND_REQ]);
        MPI_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD,
                      &my_reqs[RECV_REQ]);
    }
    else
    {
        MPI_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD,
                      &my_reqs[RECV_REQ]);
        MPI_Send_init(recv_buf, BUFFER_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD,
                      &my_reqs[SEND_REQ]);
    }

    double start = MPI_Wtime();
    for (int i = 0; i < num_iters; i++)
    {
        if (0 == rank)
        {
            // Ping side
            pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)send_buf, BUFFER_SIZE, i);
            check_hip(hipDeviceSynchronize());
            MPI_Startall(2, my_reqs);
            MPI_Waitall(2, my_reqs, MPI_STATUSES_IGNORE);
            // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
            //     (int*)recv_buf, BUFFER_SIZE, i, rank);
        }
        else
        {
            MPI_Start(&my_reqs[RECV_REQ]);
            MPI_Wait(&my_reqs[RECV_REQ], MPI_STATUS_IGNORE);
            // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
            //     (int*)recv_buf, BUFFER_SIZE, i, rank);
            pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)send_buf, (int*)recv_buf, BUFFER_SIZE);
            check_hip(hipDeviceSynchronize());
            MPI_Start(&my_reqs[SEND_REQ]);
            MPI_Wait(&my_reqs[SEND_REQ], MPI_STATUS_IGNORE);
        }
    }
    double end = MPI_Wtime();

    // Final check
    check_hip(hipDeviceSynchronize());
    print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
        (int*)recv_buf, BUFFER_SIZE, num_iters - 1, rank);
    check_hip(hipDeviceSynchronize());

    // Cleanup
    MPI_Request_free(&my_reqs[SEND_REQ]);
    MPI_Request_free(&my_reqs[RECV_REQ]);

    std::cout << rank << " is done: " << end - start << std::endl;

    MPI_Finalize();

    return 0;
}