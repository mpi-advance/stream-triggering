// [TODO] Temporary: delete once the gfx90a IPC hang check is confirmed (Frontier, 2026-11-27).
// pingpong_ipc with each ready flag written by a kernel instead of a hipMemcpyDtoDAsync: a
// system-scope release fence, the store, and a second fence to write the flag itself back.
#include <vector>

#include "benchmark.hpp"

__global__ void set_buffer(int* buffer, int value)
{
    *buffer = value;
}

__global__ void signal_peer(int* peer_flag, int value)
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __hip_atomic_store(peer_flag, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}

__global__ void wait_for_value(volatile int* location, int value)
{
    int current_value = *location;
    while (current_value != value)
    {
        current_value = *location;
    }
}

int main(int argc, char* argv[])
{
    Benchmark::init_benchmark<false, false>(&argc, &argv);

    void* ready_buffer = nullptr;
    Common::allocate_gpu_memory(&ready_buffer, sizeof(int));
    set_buffer<<<1, 1>>>((int*)ready_buffer, 0);
    device_sync();

    std::vector<hipIpcMemHandle_t> my_mem_handles(2);
    check_gpu(hipIpcGetMemHandle(&my_mem_handles[0], Benchmark::send_buffer));
    check_gpu(hipIpcGetMemHandle(&my_mem_handles[1], ready_buffer));

    std::vector<hipIpcMemHandle_t> peer_mem_handles(2);
    MPI_Sendrecv(my_mem_handles.data(), sizeof(hipIpcMemHandle_t) * 2, MPI_BYTE,
                 (Benchmark::rank + 1) % 2, 0, peer_mem_handles.data(),
                 sizeof(hipIpcMemHandle_t) * 2, MPI_BYTE, (Benchmark::rank + 1) % 2, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    void* d_peer_ptr;
    check_gpu(hipIpcOpenMemHandle(&d_peer_ptr, peer_mem_handles[0],
                                  hipIpcMemLazyEnablePeerAccess));
    void* d_peer_ready_ptr;
    check_gpu(hipIpcOpenMemHandle(&d_peer_ready_ptr, peer_mem_handles[1],
                                  hipIpcMemLazyEnablePeerAccess));

    auto do_cycles = [&](int num_cycles) {
        using namespace Benchmark;
        for (int i = 0; i < num_cycles; i++)
        {
            if (0 == rank)
            {
                pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)send_buffer, BUFFER_SIZE, i);
                signal_peer<<<1, 1, 0, bench_stream>>>((int*)d_peer_ready_ptr, i + 1);
                wait_for_value<<<1, 1, 0, bench_stream>>>((int*)ready_buffer, i + 1);
                check_gpu(hipMemcpyDtoDAsync(recv_buffer, d_peer_ptr,
                                             sizeof(int) * BUFFER_SIZE, bench_stream));
            }
            else
            {
                wait_for_value<<<1, 1, 0, bench_stream>>>((int*)ready_buffer, i + 1);
                check_gpu(hipMemcpyDtoDAsync(recv_buffer, d_peer_ptr,
                                             sizeof(int) * BUFFER_SIZE, bench_stream));
                pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)send_buffer, (int*)recv_buffer, BUFFER_SIZE);
                signal_peer<<<1, 1, 0, bench_stream>>>((int*)d_peer_ready_ptr, i + 1);
            }
        }
        stream_sync(bench_stream);
    };

    Benchmark::run_experiment(do_cycles);
    MPI_Finalize();

    check_gpu(hipIpcCloseMemHandle(d_peer_ptr));
    check_gpu(hipIpcCloseMemHandle(d_peer_ready_ptr));

    Benchmark::cleanup();
    check_gpu(hipFree(ready_buffer));

    return 0;
}
