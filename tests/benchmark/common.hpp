#ifndef BENCHMARK_COMMON
#define BENCHMARK_COMMON

#include <unistd.h>

#include <iostream>

#include <hip/hip_runtime.h>
#include "mpi.h"
#include "stream-triggering.h"

#define check_hip(function)                       \
    {                                              \
        auto err = function;                       \
        check_hip_error(err, __FILE__, __LINE__); \
    }

#define force_hip(function)                             \
    {                                                    \
        auto err = function;                             \
        check_hip_error<true>(err, __FILE__, __LINE__); \
    }

template <bool shouldThrow = false>
void check_hip_error(const hipError_t err, const char* filename,
                      const int line)
{
    if (err != hipSuccess)
    {
        std::cout << "(" << err << ") in " << filename << " on line " << line
                  << " : " << hipGetErrorString(err) << std::endl;
        if constexpr (shouldThrow)
        {
            throw std::runtime_error(hipGetErrorString(err));
        }
    }
}

__global__ void init_buffers(int* send_buf, int* recv_buf, int buffer_len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    send_buf[index] = 0;
    recv_buf[index] = -1;
}

__global__ void pack_buffer(int* buffer, int buffer_len, int iteration)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    buffer[index] = iteration * 100;
}

__global__ void pack_buffer2(int* buffer, int* recvd_buffer, int buffer_len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    buffer[index] = recvd_buffer[index];
}

__global__ void print_buffer(volatile int* buffer, int buffer_len,
                             int iteration, int rank)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    if (buffer[index] != iteration * 100)
    {
        printf("<GPU %d> Wrong buffer value @ index: %d Got: %d Expected: %d\n",
               rank, index, buffer[index], iteration * 100);
    }
}

static void inline check_param_size(int* argc, int num_params, std::string usage)
{
    if((*argc) != (1+num_params))
    {
        std::cerr << "Usage: "<< usage << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

static void inline read_iter_buffer_input(char*** argv, int* num_iters, int* buffer_size)
{
    (*num_iters) = std::atoi((*argv)[1]);
    (*buffer_size) = std::atoi((*argv)[2]);
}


#endif
