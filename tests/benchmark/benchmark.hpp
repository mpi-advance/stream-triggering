#include "../common/common.hpp"

#if defined(USE_STREAM_TRIGGERING)
#include "stream-triggering.h"
#endif

namespace Benchmark
{
int rank;

int num_warmups;
int num_iters;
int BUFFER_SIZE;
int BLOCK_SIZE;
int NUM_BLOCKS;

void* send_buffer;
void* recv_buffer;

#if defined(USE_STREAM_TRIGGERING)
MPI_Info mem_info;
MPIS_Queue my_queue;
#endif

#if defined(NEED_HIP)
hipStream_t bench_stream;
#elif defined(NEED_CUDA)
cudaStream_t bench_stream;
#endif

void allocate_gpu_memory(void** location, size_t size)
{
#ifndef FINE_GRAINED_TEST
#if defined(NEED_HIP)
    force_gpu(hipMalloc(location, size));
#elif defined(NEED_CUDA)
    force_gpu(cudaMalloc(location, size));
#else
    static_assert(false, "Unable to build benchmark, no valid GPU flags set.");
#endif
#else
#if defined(NEED_HIP)
    force_gpu(hipExtMallocWithFlags(location, size, hipDeviceMallocFinegrained));
#else
    static_assert(false, "Unable to build benchmark for non-HIP fine grained test");
#endif
#endif
}

template <bool USE_THREAD, bool DOUBLE_BUFF>
void init_benchmark(int* argc, char*** argv)
{
    if constexpr (USE_THREAD)
    {
        int mode;
        MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &mode);
    }
    else
    {
        MPI_Init(argc, argv);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // I want "two params"
    check_param_size(argc, 2, "<program> <number of iterations> <buffer size>");

    // Input parameters
    num_warmups = 10;
    num_iters   = 0;
    BUFFER_SIZE = 0;
    read_iter_buffer_input(argv, &num_iters, &BUFFER_SIZE);

    // Make Buffers
    BLOCK_SIZE         = 128;
    NUM_BLOCKS         = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int scaling_factor = DOUBLE_BUFF + 1;
    // > Main pingpong buffers
    send_buffer = nullptr;
    recv_buffer = nullptr;
#if defined(USE_STREAM_TRIGGERING)
    // Info hint for buffer
    MPI_Info_create(&mem_info);
#ifndef FINE_GRAINED_TEST
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:coarse");
#else
    MPI_Info_set(mem_info, "mpi_memory_alloc_kinds", "rocm:device:fine");
#endif
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &send_buffer);
    MPIS_Alloc_mem(sizeof(int) * BUFFER_SIZE, mem_info, &recv_buffer);
#else
    allocate_gpu_memory(&send_buffer, sizeof(int) * BUFFER_SIZE * scaling_factor);
    allocate_gpu_memory(&recv_buffer, sizeof(int) * BUFFER_SIZE * scaling_factor);
#endif

    init_buffers<<<NUM_BLOCKS, BLOCK_SIZE>>>((int*)send_buffer, (int*)recv_buffer,
                                             BUFFER_SIZE * scaling_factor);
    device_sync();

#if defined(NEED_HIP)
    check_gpu(hipStreamCreateWithFlags(&bench_stream, hipStreamNonBlocking));
#elif defined(NEED_CUDA)
    check_gpu(cudaStreamCreateWithFlags(&bench_stream, cudaStreamNonBlocking));
#endif

    // Make queue, if using stream triggering
#if defined(USE_STREAM_TRIGGERING)
#if defined(HIP_BACKEND)
    MPIS_Queue_init(&my_queue, GPU_MEM_OPS, &Benchmark::bench_stream);
#elif defined(CUDA_BACKEND)
    MPIS_Queue_init(&my_queue, GPU_MEM_OPS, &Benchmark::bench_stream);
#elif defined(CXI_BACKEND)
    MPIS_Queue_init(&my_queue, CXI, &Benchmark::bench_stream);
#elif defined(THREAD_BACKEND)
    MPIS_Queue_init(&my_queue, THREAD, &Benchmark::bench_stream);
#endif
#endif
}

template <typename LambdaFxn>
void run_experiment(LambdaFxn do_cycles, bool is_double_buffered = false)
{
    // Carry out some warmup runs to get things started.
    MPI_Barrier(MPI_COMM_WORLD);
    do_cycles(num_warmups);
    // Do performance run
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    do_cycles(num_iters);
    double end = MPI_Wtime();

    // Final check
    device_sync();
    if (is_double_buffered && 0 == (num_iters % 2))
    {
        // Cast to char* to do pointer math, then back to int* for function
        print_buffer<<<1, BLOCK_SIZE, 0, bench_stream>>>(
            (int*)((char*)recv_buffer + (sizeof(int) * BUFFER_SIZE)), BUFFER_SIZE,
            num_iters - 1, rank);
    }
    else
    {
        print_buffer<<<1, BLOCK_SIZE, 0, bench_stream>>>((int*)recv_buffer, BUFFER_SIZE,
                                                         num_iters - 1, rank);
    }
    device_sync();

    std::cout << rank << " is done: " << end - start << std::endl;
}

void cleanup()
{
#if defined(USE_STREAM_TRIGGERING)
    MPIS_Free_mem(send_buffer);
    MPIS_Free_mem(recv_buffer);
    MPI_Info_free(&mem_info);
    MPIS_Queue_free(&my_queue);
#else
#if defined(NEED_HIP)
    check_gpu(hipFree(send_buffer));
    check_gpu(hipFree(recv_buffer));
#elif defined(NEED_CUDA)
    check_gpu(cudaFree(send_buffer));
    check_gpu(cudaFree(recv_buffer));
#endif
#endif
}
}  // namespace Benchmark