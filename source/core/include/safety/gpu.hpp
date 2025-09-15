#ifndef ST_GPU_SAFETY
#define ST_GPU_SAFETY

#ifdef HIP_GPUS
#include "hip/hip_runtime.h"
#endif
#ifdef CUDA_GPUS
#include "cuda.h"
#endif

#include <iostream>

#define check_gpu(function)                       \
    {                                             \
        auto err = function;                      \
        check_gpu_error(err, __FILE__, __LINE__); \
    }

#define force_gpu(function)                             \
    {                                                   \
        auto err = function;                            \
        check_gpu_error<true>(err, __FILE__, __LINE__); \
    }

#ifdef HIP_GPUS
template <bool shouldThrow = false>
void check_gpu_error(const hipError_t err, const char* filename, const int line)
{
    if (err != hipSuccess)
    {
        std::cout << "(" << err << ") in " << filename << " on line " << line << " : "
                  << hipGetErrorString(err) << std::endl;
        if constexpr (shouldThrow)
        {
            throw std::runtime_error(hipGetErrorString(err));
        }
    }
}
#endif

#ifdef CUDA_GPUS
template <bool shouldThrow = false>
void check_gpu_error(const cudaError_t err, const char* filename, const int line)
{
    if (err != cudaSuccess)
    {
        std::cout << "(" << err << ") in " << filename << " on line " << line << " : "
                  << cudaGetErrorString(err) << std::endl;
        if constexpr (shouldThrow)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
}

template <bool shouldThrow = false>
void check_gpu_error(const CUresult code, const char* filename, const int line)
{
    if (code != CUDA_SUCCESS)
    {
        const char* name = nullptr;
        const char* strg = nullptr;

        cuGetErrorName(code, &name);
        cuGetErrorString(code, &strg);

        std::cout << "(" << code << ") in " << filename << " on line " << line << " : "
                  << std::string(name) << " - " << std::string(strg) << std::endl;
        if constexpr (shouldThrow)
        {
            throw std::runtime_error("CU CUDA Error");
        }
    }
}
#endif

#endif