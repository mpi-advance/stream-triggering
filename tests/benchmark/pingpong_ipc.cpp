#include <vector>

#include "benchmark.hpp"

__global__ void set_buffer(int* buffer, int value)
{
    *buffer = value;
}

__global__ void set_buffers(int* read_buff, int* compl_buff, int compl_value)
{
    *read_buff  = 0;
    *compl_buff = compl_value;
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

    // > Buffers for completion notifications
    void* ready_buffer       = nullptr;
    void* ready_value_buffer = nullptr;
    Benchmark::allocate_gpu_memory(&ready_buffer, sizeof(int));
    Benchmark::allocate_gpu_memory(&ready_value_buffer, sizeof(int));

    set_buffers<<<1, 1>>>((int*)ready_buffer, (int*)ready_value_buffer, 1);
    device_sync();

    // Exchange HipIPC information
    // > Make memory handle for both send buffer and ready buffer
    std::vector<hipIpcMemHandle_t> my_mem_handles(2);
    check_gpu(hipIpcGetMemHandle(&my_mem_handles[0], Benchmark::send_buffer));
    check_gpu(hipIpcGetMemHandle(&my_mem_handles[1], ready_buffer));

    // > Get partner's handles
    std::vector<hipIpcMemHandle_t> peer_mem_handles(2);
    MPI_Sendrecv(my_mem_handles.data(), sizeof(hipIpcMemHandle_t) * 2, MPI_BYTE,
                 (Benchmark::rank + 1) % 2, 0, peer_mem_handles.data(),
                 sizeof(hipIpcMemHandle_t) * 2, MPI_BYTE, (Benchmark::rank + 1) % 2, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // > Open handle to peer's data buffer
    void* d_peer_ptr;
    check_gpu(hipIpcOpenMemHandle(&d_peer_ptr, peer_mem_handles[0],
                                  hipIpcMemLazyEnablePeerAccess));
    // > Open handle to peer's ready buffer
    void* d_peer_ready_ptr;
    check_gpu(hipIpcOpenMemHandle(&d_peer_ready_ptr, peer_mem_handles[1],
                                  hipIpcMemLazyEnablePeerAccess));

    // Main pingpong lambda function
    auto do_cycles = [&](int num_cycles) {
        using namespace Benchmark;
        for (int i = 0; i < num_cycles; i++)
        {
            // Prepare the "ready" value, both sides can do this first.
            set_buffer<<<1, 1, 0, bench_stream>>>((int*)ready_value_buffer, i + 1);

            if (0 == rank)  // Ping side
            {
                // > Pack data to send
                pack_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)send_buffer, BUFFER_SIZE, i);
                // > Move the ready buffer to peer
                check_gpu(hipMemcpyDtoDAsync(d_peer_ready_ptr, ready_value_buffer,
                                             sizeof(int), bench_stream));
                // > Wait for peer to finish packing
                wait_for_value<<<1, 1, 0, bench_stream>>>((int*)ready_buffer, i + 1);
                // > Get data back
                check_gpu(hipMemcpyDtoDAsync(recv_buffer, d_peer_ptr,
                                             sizeof(int) * BUFFER_SIZE, bench_stream));
                // Verify
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                //    (int*)recv_buffer, BUFFER_SIZE, i, rank);
            }
            else  // Pong side
            {
                // > Wait for other side to have the data ready for us
                wait_for_value<<<1, 1, 0, bench_stream>>>((int*)ready_buffer, i + 1);
                // > Get the data from them
                check_gpu(hipMemcpyDtoDAsync(recv_buffer, d_peer_ptr,
                                             sizeof(int) * BUFFER_SIZE, bench_stream));
                // > Verify
                // print_buffer<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                //     (int*)recv_buffer, BUFFER_SIZE, i, rank);
                // > Pack the data to be sent back (aka compute)
                pack_buffer2<<<NUM_BLOCKS, BLOCK_SIZE, 0, bench_stream>>>(
                    (int*)send_buffer, (int*)recv_buffer, BUFFER_SIZE);
                // > Tell the other side we are ready
                check_gpu(hipMemcpyDtoDAsync(d_peer_ready_ptr, ready_value_buffer,
                                             sizeof(int), bench_stream));
            }
        }
        stream_sync(bench_stream);
    };

    Benchmark::run_experiment(do_cycles);
    MPI_Finalize();

    // Hip cleanup
    check_gpu(hipIpcCloseMemHandle(d_peer_ptr));
    check_gpu(hipIpcCloseMemHandle(d_peer_ready_ptr));

    Benchmark::cleanup();
    check_gpu(hipFree(ready_buffer));
    check_gpu(hipFree(ready_value_buffer));

    return 0;
}