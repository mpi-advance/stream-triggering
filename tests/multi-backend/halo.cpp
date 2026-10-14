#include "../common/checks.hpp"
#include "stream-triggering.h"

__global__ void init_halos(int* top_s, int* top_r, int* bottom_s, int* bottom_r,
                           int* left_s, int* left_r, int* right_s, int* right_r,
                           int buffer_len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    top_s[index]    = -1;
    top_r[index]    = -1;
    bottom_s[index] = -1;
    bottom_r[index] = -1;
    left_s[index]   = -1;
    left_r[index]   = -1;
    right_s[index]  = -1;
    right_r[index]  = -1;
}

__global__ void pack_halos(int* top_s, int* bottom_s, int* left_s, int* right_s,
                           int buffer_len, int rank, int iter)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    top_s[index]    = (rank * 100) + iter;
    bottom_s[index] = (rank * 100) + iter;
    left_s[index]   = (rank * 100) + iter;
    right_s[index]  = (rank * 100) + iter;
}

int main(int argc, char* argv[])
{
    using namespace Checks;

    // Input parameters
    int num_iters   = Checks::iterations(10);
    int BUFFER_SIZE = Checks::buffer_size(64);
    int NUM_BLOCKS  = Common::num_blocks(BUFFER_SIZE);

    // Memory info, stream and queue (setup()'s send/recv buffers go unused)
    Checks::setup(&argc, &argv, BUFFER_SIZE, 1);

    // Make Buffers

    void* top_s    = nullptr;
    void* top_r    = nullptr;
    void* bottom_s = nullptr;
    void* bottom_r = nullptr;
    void* left_s   = nullptr;
    void* left_r   = nullptr;
    void* right_s  = nullptr;
    void* right_r  = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &top_s);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &top_r);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &bottom_s);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &bottom_r);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &left_s);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &left_r);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &right_s);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &right_r);

    init_halos<<<NUM_BLOCKS, Common::BLOCK_SIZE>>>(
        (int*)top_s, (int*)top_r, (int*)bottom_s, (int*)bottom_r, (int*)left_s,
        (int*)left_r, (int*)right_s, (int*)right_r, BUFFER_SIZE);
    device_sync();

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int      periods[2] = {true, true};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &cart_comm);

    int rank2;
    MPI_Comm_rank(cart_comm, &rank2);

    // Find neighbors with Cartesian shifts: directions 0=x, 1=y
    int nbr_left, nbr_right, nbr_down, nbr_up;
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_left, &nbr_right);  // shift along y (cols)
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_down, &nbr_up);     // shift along x (rows)

    MPI_Group base_group;
    MPI_Comm_group(cart_comm, &base_group);
    MPI_Group lookup_group;
    MPI_Comm_group(MPI_COMM_WORLD, &lookup_group);
    int base_ranks[4]   = {nbr_up, nbr_down, nbr_left, nbr_right};
    int lookup_ranks[4] = {-1, -1, -1, -1};
    MPI_Group_translate_ranks(base_group, 4, base_ranks, lookup_group, lookup_ranks);
    MPI_Group_free(&base_group);
    MPI_Group_free(&lookup_group);

    // Neighbors as "cart rank(world rank)": up, down, left, right
    if (Checks::verbose)
    {
        std::cout << rank << "(" << rank2 << "): " << nbr_up << "(" << lookup_ranks[0]
                  << ") " << nbr_down << "(" << lookup_ranks[1] << ") " << nbr_left << "("
                  << lookup_ranks[2] << ") " << nbr_right << "(" << lookup_ranks[3] << ")"
                  << std::endl;
    }

    // Each halo holds (100 * sender's rank) + iteration
    void* recv_halos[4] = {top_r, bottom_r, left_r, right_r};
    auto  check_halos   = [&](int iter) {
        for (int k = 0; k < 4; k++)
        {
            Checks::expect_buffer(my_stream, recv_halos[k], BUFFER_SIZE,
                                     (100 * lookup_ranks[k]) + iter, 0);
        }
    };

    // Make requests
    MPIS_Request my_reqs[8];

    // Left
    MPIS_Recv_init(left_r, BUFFER_SIZE, MPI_INT, nbr_left, 50, cart_comm, mem_info,
                   &my_reqs[0]);
    // Right
    MPIS_Recv_init(right_r, BUFFER_SIZE, MPI_INT, nbr_right, 50, cart_comm, mem_info,
                   &my_reqs[1]);
    // Top
    MPIS_Recv_init(top_r, BUFFER_SIZE, MPI_INT, nbr_up, 50, cart_comm, mem_info,
                   &my_reqs[2]);
    // Bottom
    MPIS_Recv_init(bottom_r, BUFFER_SIZE, MPI_INT, nbr_down, 50, cart_comm, mem_info,
                   &my_reqs[3]);

    MPIS_Send_init(left_s, BUFFER_SIZE, MPI_INT, nbr_left, 50, cart_comm, mem_info,
                   &my_reqs[4]);
    MPIS_Send_init(right_s, BUFFER_SIZE, MPI_INT, nbr_right, 50, cart_comm, mem_info,
                   &my_reqs[5]);
    MPIS_Send_init(top_s, BUFFER_SIZE, MPI_INT, nbr_up, 50, cart_comm, mem_info,
                   &my_reqs[6]);
    MPIS_Send_init(bottom_s, BUFFER_SIZE, MPI_INT, nbr_down, 50, cart_comm, mem_info,
                   &my_reqs[7]);

    Common::match_in_place(8, my_reqs);
    MPI_Barrier(MPI_COMM_WORLD);

    auto do_cycles = [&](int num_cycles) {
        for (int i = 0; i < num_cycles; i++)
        {
            // Post Receives
            MPIS_Enqueue_startall(my_queue, 4, my_reqs);
            // Packing Kernel
            pack_halos<<<NUM_BLOCKS, Common::BLOCK_SIZE, 0, my_stream>>>(
                (int*)top_s, (int*)bottom_s, (int*)left_s, (int*)right_s, BUFFER_SIZE,
                rank, i);

#ifdef THREAD_BACKEND
            device_sync();
#endif

            MPIS_Enqueue_startall(my_queue, 4, &my_reqs[4]);
            MPIS_Enqueue_waitall(my_queue, 8, my_reqs, MPI_STATUSES_IGNORE);

#ifdef THREAD_BACKEND
            MPIS_Queue_wait(&my_queue);
#endif
            check_halos(i);
#ifdef THREAD_BACKEND
            // The next receives start right away, so finish checking these first
            device_sync();
#endif
            if (Checks::verbose)
                std::cout << rank << " step: " << i << "/" << num_iters << std::endl;
        }
        MPIS_Queue_wait(&my_queue);
    };

    double start = MPI_Wtime();
    do_cycles(num_iters);
    double end = MPI_Wtime();
    // Final check
    check_halos(num_iters - 1);
    device_sync();  // Before the halo buffers are freed

    // Cleanup
    MPIS_Request_freeall(8, my_reqs);
    MPIS_Free_mem(top_s);
    MPIS_Free_mem(top_r);
    MPIS_Free_mem(bottom_s);
    MPIS_Free_mem(bottom_r);
    MPIS_Free_mem(left_s);
    MPIS_Free_mem(left_r);
    MPIS_Free_mem(right_s);
    MPIS_Free_mem(right_r);

    MPI_Comm_free(&cart_comm);

    if (Checks::verbose)
        std::cout << rank << " is done: " << end - start << std::endl;

    return Checks::teardown("halo");
}