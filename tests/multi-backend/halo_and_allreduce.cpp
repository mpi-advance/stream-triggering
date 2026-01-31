#include "../common/common.hpp"
#include "../common/timers.hpp"
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

__global__ void print_halos(int* top_r, int* bottom_r, int* left_r, int* right_r,
                            int buffer_len, int rank, int top_peer, int bottom_peer,
                            int left_peer, int right_peer, int iter)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    if (((top_peer * 100) + iter) != top_r[index])
    {
        printf("<GPU %d> Top: %d \n", rank, top_r[index]);
    }
    if (((bottom_peer * 100) + iter) != bottom_r[index])
    {
        printf("<GPU %d> Bottom: %d \n", rank, bottom_r[index]);
    }
    if (((left_peer * 100) + iter) != left_r[index])
    {
        printf("<GPU %d> Left: %d \n", rank, left_r[index]);
    }
    if (((right_peer * 100) + iter) != right_r[index])
    {
        printf("<GPU %d> Right: %d \n", rank, right_r[index]);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    // int thread_mode;
    // MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_mode);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // I want "two params"
    check_param_size(&argc, 2, "<program> <number of iterations> <local buffer size>");

    // Input parameters
    int num_warmups = 10;
    int num_iters   = 0;
    int BUFFER_SIZE = 0;
    read_iter_buffer_input(&argv, &num_iters, &BUFFER_SIZE);

    // Info hint for buffer
    MPI_Info mem_info;
    MPI_Info_create(&mem_info);
#ifndef FINE_GRAINED_TEST
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:coarse");
#else
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:fine");
#endif

    // Make Buffers
    int BLOCK_SIZE = 128;
    int NUM_BLOCKS = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

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

    init_halos<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)top_s, (int*)top_r, (int*)bottom_s,
                                           (int*)bottom_r, (int*)left_s, (int*)left_r,
                                           (int*)right_s, (int*)right_r, BUFFER_SIZE);

    /* Init memory for allreduce */
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &send_buf);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &recv_buf);
    init_buffers2<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buf, (int*)recv_buf, BUFFER_SIZE,
                                              rank);

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

    std::string rank_patterns = std::to_string(rank);
    rank_patterns += "(";
    rank_patterns += std::to_string(rank2);
    rank_patterns += "): ";
    rank_patterns += std::to_string(nbr_up);
    rank_patterns += "(";
    rank_patterns += std::to_string(lookup_ranks[0]);
    rank_patterns += ") ";
    rank_patterns += std::to_string(nbr_down);
    rank_patterns += "(";
    rank_patterns += std::to_string(lookup_ranks[1]);
    rank_patterns += ") ";
    rank_patterns += std::to_string(nbr_left);
    rank_patterns += "(";
    rank_patterns += std::to_string(lookup_ranks[2]);
    rank_patterns += ") ";
    rank_patterns += std::to_string(nbr_right);
    rank_patterns += "(";
    rank_patterns += std::to_string(lookup_ranks[3]);
    rank_patterns += ") ";

    std::cout << rank_patterns << std::endl;

#if defined(NEED_HIP)
    hipStream_t my_stream;
    check_gpu(hipStreamCreateWithFlags(&my_stream, hipStreamNonBlocking));
#elif defined(NEED_CUDA)
    cudaStream_t my_stream;
    check_gpu(cudaStreamCreateWithFlags(&my_stream, cudaStreamNonBlocking));
#endif

    // Make queue
    MPIS_Queue my_queue;
#if defined(HIP_BACKEND)
    MPIS_Queue_init(&my_queue, GPU_MEM_OPS, &my_stream);
#elif defined(CUDA_BACKEND)
    MPIS_Queue_init(&my_queue, GPU_MEM_OPS, &my_stream);
#elif defined(CXI_BACKEND)
    MPIS_Queue_init(&my_queue, CXI, &my_stream);
#elif defined(THREAD_BACKEND)
    MPIS_Queue_init(&my_queue, THREAD, &my_stream);
#endif

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

    MPIS_Matchall(8, my_reqs, MPI_STATUSES_IGNORE);

    MPIS_Request my_req;
    MPIS_Allreduce_init(send_buf, recv_buf, BUFFER_SIZE, MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                        MPI_INFO_NULL, &my_req);
    MPIS_Queue_match(my_queue, &my_req, MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);

    auto do_cycles = [&](int num_cycles) {
        for (int i = 0; i < num_cycles; i++)
        {
            // Post Receives
            MPIS_Enqueue_startall(my_queue, 4, my_reqs);
            // Packing Kernel
            pack_halos<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)top_s, (int*)bottom_s, (int*)left_s, (int*)right_s, BUFFER_SIZE,
                rank, i);
            init_buffers2<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)send_buf, (int*)recv_buf, BUFFER_SIZE, rank);

            // #ifdef THREAD_BACKEND
            //             device_sync();
            // #endif

            MPIS_Enqueue_startall(my_queue, 4, &my_reqs[4]);
            MPIS_Enqueue_waitall(my_queue);
            MPIS_Enqueue_start(my_queue, &my_req);
            MPIS_Enqueue_waitall(my_queue);

            // #ifdef THREAD_BACKEND
            // MPIS_Queue_wait(my_queue);
            // #endif
            print_halos<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
                (int*)top_r, (int*)bottom_r, (int*)left_r, (int*)right_r, BUFFER_SIZE,
                rank, lookup_ranks[0], lookup_ranks[1], lookup_ranks[2], lookup_ranks[3],
                i);
            verify<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)recv_buf, BUFFER_SIZE,
                                                             rank, size);
            std::cout << rank << " step: " << i << "/" << num_iters << std::endl;
        }
        MPIS_Queue_wait(my_queue);
    };

    double start = MPI_Wtime();
    do_cycles(num_iters);
    double end = MPI_Wtime();
    // Final check
    std::cout << rank << " done queueing, about to sync." << std::endl;
    device_sync();
    print_halos<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>(
        (int*)top_r, (int*)bottom_r, (int*)left_r, (int*)right_r, BUFFER_SIZE, rank,
        lookup_ranks[0], lookup_ranks[1], lookup_ranks[2], lookup_ranks[3],
        num_iters - 1);
    verify<<<NUM_BLOCKS, BLOCK_SIZE, 0, my_stream>>>((int*)recv_buf, BUFFER_SIZE, rank,
                                                     size);
    device_sync();

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

    // Allreduce cleanup
    MPIS_Request_freeall(1, &my_req);
    MPIS_Free_mem(send_buf);
    MPIS_Free_mem(recv_buf);

    MPI_Info_free(&mem_info);

    MPIS_Queue_free(&my_queue);

    MPI_Comm_free(&cart_comm);

    std::cout << rank << " is done: " << end - start << std::endl;

    MPI_Finalize();

    return 0;
}