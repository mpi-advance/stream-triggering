#include "../common/checks.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    Checks::init(&argc, &argv);

    int rank = Checks::rank;
    int peer = rank ^ 1;

    // Input parameters
    int num_warmups = 10;
    int num_iters   = Checks::iterations(10);
    int BUFFER_SIZE = Checks::buffer_size(64);

    // Make Buffers
    int NUM_BLOCKS = Common::num_blocks(BUFFER_SIZE);

    // Info hint for buffer
    MPI_Info mem_info = Common::make_gpu_mem_info();

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE * 2, mem_info, &send_buf);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE * 2, mem_info, &recv_buf);

    init_buffers<<<Common::num_blocks(BUFFER_SIZE * 2), Common::BLOCK_SIZE>>>(
        (int*)send_buf, (int*)recv_buf, BUFFER_SIZE * 2);
    device_sync();

    StreamType my_stream = Setup::create_stream();

    // Make queue
    MPIS_Queue my_queue;
    Setup::create_queue(&my_queue, &my_stream);

#define SEND_REQ ((rank & 1) ^ 1)
#define RECV_REQ (rank & 1)

    // Requests for both halves of the buffers, alternated between iterations
    MPIS_Request my_reqs[2];
    MPIS_Request my_other_reqs[2];
    // MPIS_Request queue_reqs[2]; TODO
    int offset = sizeof(int) * BUFFER_SIZE;

    void*         active_send_buffer;
    void*         active_recv_buffer;
    void*         inactive_send_buffer;
    void*         inactive_recv_buffer;
    MPIS_Request* active_request_ptr;
    MPIS_Request* inactive_request_ptr;

    // Creates and matches the requests, and makes the first half active. The odd rank
    // echoes straight from its receive buffers.
    auto make_requests = [&]() {
        if (0 == rank % 2)
        {
            MPIS_Send_init(send_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD,
                           mem_info, &my_reqs[SEND_REQ]);
            MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD,
                           mem_info, &my_reqs[RECV_REQ]);
            MPIS_Send_init((char*)send_buf + offset, BUFFER_SIZE, MPI_INT, peer, 0,
                           MPI_COMM_WORLD, mem_info, &my_other_reqs[SEND_REQ]);
            MPIS_Recv_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, peer, 0,
                           MPI_COMM_WORLD, mem_info, &my_other_reqs[RECV_REQ]);
        }
        else
        {
            MPIS_Recv_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD,
                           mem_info, &my_reqs[RECV_REQ]);
            MPIS_Send_init(recv_buf, BUFFER_SIZE, MPI_INT, peer, 0, MPI_COMM_WORLD,
                           mem_info, &my_reqs[SEND_REQ]);
            MPIS_Recv_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, peer, 0,
                           MPI_COMM_WORLD, mem_info, &my_other_reqs[RECV_REQ]);
            MPIS_Send_init((char*)recv_buf + offset, BUFFER_SIZE, MPI_INT, peer, 0,
                           MPI_COMM_WORLD, mem_info, &my_other_reqs[SEND_REQ]);
        }

        Common::match_in_place(2, my_reqs);
        Common::match_in_place(2, my_other_reqs);
        MPI_Barrier(MPI_COMM_WORLD);

        active_send_buffer   = send_buf;
        active_recv_buffer   = recv_buf;
        inactive_send_buffer = (char*)send_buf + offset;
        inactive_recv_buffer = (char*)recv_buf + offset;
        active_request_ptr   = my_reqs;
        inactive_request_ptr = my_other_reqs;
    };

    auto do_cycles = [&](int num_cycles) {
        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == rank % 2)
            {
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(&my_queue);
#endif
                // Ping side
                pack_buffer<<<NUM_BLOCKS, Common::BLOCK_SIZE, 0, my_stream>>>(
                    (int*)active_send_buffer, BUFFER_SIZE, i);
#ifdef THREAD_BACKEND
                device_sync();
#endif
                MPIS_Enqueue_startall(my_queue, 2, active_request_ptr);
                MPIS_Enqueue_waitall(my_queue, 2, active_request_ptr,
                                     MPI_STATUSES_IGNORE);
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(&my_queue);
#endif
                Checks::expect_buffer(my_stream, active_recv_buffer, BUFFER_SIZE, i * 100,
                                      0);
            }
            else
            {
                MPIS_Enqueue_start(my_queue, &active_request_ptr[RECV_REQ]);
                MPIS_Enqueue_wait(my_queue, &active_request_ptr[RECV_REQ],
                                  MPI_STATUS_IGNORE);
#ifdef THREAD_BACKEND
                MPIS_Queue_wait(&my_queue);
#endif
                Checks::expect_buffer(my_stream, active_recv_buffer, BUFFER_SIZE, i * 100,
                                      0);
#ifdef THREAD_BACKEND
                device_sync();
#endif
                MPIS_Enqueue_start(my_queue, &active_request_ptr[SEND_REQ]);
                MPIS_Enqueue_wait(my_queue, &active_request_ptr[SEND_REQ],
                                  MPI_STATUS_IGNORE);
            }

            void* temp_send      = active_send_buffer;
            active_send_buffer   = inactive_send_buffer;
            inactive_send_buffer = temp_send;

            void* temp_recv      = active_recv_buffer;
            active_recv_buffer   = inactive_recv_buffer;
            inactive_recv_buffer = temp_recv;

            MPIS_Request* temp_reqs = active_request_ptr;
            active_request_ptr      = inactive_request_ptr;
            inactive_request_ptr    = temp_reqs;
        }

        MPIS_Queue_wait(&my_queue);
    };

    make_requests();
    do_cycles(num_warmups);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    do_cycles(num_iters);
    double end = MPI_Wtime();

    // Final check: the last iteration's data
    Checks::expect_buffer(my_stream, inactive_recv_buffer, BUFFER_SIZE,
                          (num_iters - 1) * 100, 0);
    device_sync();  // Before the buffers are reused or freed

    // Cleanup of first queue
    MPIS_Request_freeall(2, my_reqs);
    MPIS_Request_freeall(2, my_other_reqs);
    MPIS_Queue_free(&my_queue);

    if (Checks::verbose)
        std::cout << rank << " is done: " << end - start << std::endl;

    // Same again on a new queue
    Setup::create_queue(&my_queue, &my_stream);
    make_requests();
    if (Checks::verbose)
        std::cout << rank << " is done with second request creation and match"
                  << std::endl;

    do_cycles(num_warmups);
    if (Checks::verbose)
        std::cout << rank << " is done with second warmup" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    do_cycles(num_iters);
    end = MPI_Wtime();

    if (Checks::verbose)
        std::cout << rank << " is done with second test" << std::endl;

    // Final check: the last iteration's data
    Checks::expect_buffer(my_stream, inactive_recv_buffer, BUFFER_SIZE,
                          (num_iters - 1) * 100, 0);
    device_sync();  // Before the buffers are reused or freed

    if (Checks::verbose)
        std::cout << rank << " is done done: " << end - start << std::endl;

    // Cleanup of second time
    MPIS_Request_freeall(2, my_reqs);
    MPIS_Request_freeall(2, my_other_reqs);
    MPIS_Queue_free(&my_queue);
    MPIS_Free_mem(send_buf);
    MPIS_Free_mem(recv_buf);
    MPI_Info_free(&mem_info);

    return Checks::finish("two_queue");
}
