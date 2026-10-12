#ifndef ST_GPU_SAFETY
#define ST_GPU_SAFETY

#ifdef HIP_GPUS
#include "hip/hip_runtime.h"
#endif
#ifdef CUDA_GPUS
#include "cuda.h"
#include "cuda_runtime.h"
#endif

#include <iostream>

/**
 * @file gpu.hpp
 * @brief Helper macros and functions for ensuring proper error checking on GPU functions
 * @details This file has the classic macros for checking the return code of GPU
 * functions, as well as a function for checking the state of the last launched GPU
 * kernel. Note that the function names are GPU vendor independent; any macro should
 * compile to the correct vendor. Currently building both GPU vendors at the same time is
 * not supported.
 */

/**
 * \defgroup safety Error Checking Functions
 * @brief Any function that helps check the return codes of a library.
 * @details Often times these functions have two modes: just printing the non-success
 * error code or raising an exception when a non-success error code is given. "Success"
 * value depends on library.
 */

/** @brief Call function and print out if error code is not success.*/
#define check_gpu(function)                       \
    {                                             \
        auto err = function;                      \
        check_gpu_error(err, __FILE__, __LINE__); \
    }

/** @brief Call function and raise exception if error code is not success.*/
#define force_gpu(function)                             \
    {                                                   \
        auto err = function;                            \
        check_gpu_error<true>(err, __FILE__, __LINE__); \
    }

#ifdef HIP_GPUS
/**
 * @brief This function prints out if the error code give was not hipSuccess.
 * @details If the error code is not hipSuccess, print out the error code, the filename,
 * the line number of the function call, along with the error string of the hip error
 * code. If the template parameter is set to true, also raise an exception. Currently,
 * this exception is not tied to MPISException.
 *
 * @ingroup safety
 * @tparam shouldThrow If true, throw on non-hipSuccess instead of just printing out.
 * @param [in] err The error code returned by the HIP function
 * @param [in] filename The name of the file where the function was called.
 * @param [in] line The line number containing the code
 */
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

/** @brief Checks the return code from hipGetLastError. */
inline void check_kernel()
{
    check_gpu(hipGetLastError());
}

/** @brief Raises exception if the return code from hipGetLastError is not hipSuccess. */
inline void force_kernel()
{
    force_gpu(hipGetLastError());
}
#endif

#ifdef CUDA_GPUS
/**
 * @brief This function prints out if the error code give was not cudaSuccess.
 * @details If the error code is not cudaSuccess, print out the error code, the filename,
 * the line number of the function call, along with the error string of the CUDA error
 * code. If the template parameter is set to true, also raise an exception. Currently,
 * this exception is not tied to MPISException.
 *
 * @ingroup safety
 * @tparam shouldThrow If true, throw on non-cudaSuccess instead of just printing out.
 * @param [in] err The error code returned by the CUDA function
 * @param [in] filename The name of the file where the function was called.
 * @param [in] line The line number containing the code
 */
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

/**
 * @brief This function checks the value of CUDA runtime return code.
 * @details If the error code is not CUDA_SUCCESS, print out the error code, the filename,
 * the line number of the function call, along with the error string of the CUDA error
 * code. If the template parameter is set to true, also raise an exception. Currently,
 * this exception is not tied to MPISException.
 *
 * @tparam shouldThrow If true, throw on non-CUDA_SUCCESS instead of just printing out.
 * @param [in] code The error code returned by the CUDA function
 * @param [in] filename The name of the file where the function was called.
 * @param [in] line The line number containing the code
 */
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

/** @brief Checks the return code from cudaGetLastError. */
inline void check_kernel()
{
    check_gpu(cudaGetLastError());
}

/** @brief Raises exception if the return code from cudaGetLastError is not cudaSuccess.
 */
inline void force_kernel()
{
    force_gpu(cudaGetLastError());
}

#endif

#endif