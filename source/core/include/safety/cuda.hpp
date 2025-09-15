#ifndef ST_CUDA_SAFETY
#define ST_CUDA_SAFETY

#include "cuda.h"

#include <iostream>

/** @brief wrapper around Cuda functions to note errors and continue 
 * @ingroup debug 
 */
#define check_cuda(function)                                                                        \
	{                                                                                               \
		auto err = function;                                                                        \
		check_cuda_error(err, __FILE__, __LINE__);                                                  \
	}

/** @brief wrapper around Cuda functions to note errors and continue 
* @ingroup debug 
*/
#define force_cuda(function)                                                                        \
	{                                                                                               \
		auto err = function;                                                                        \
		check_cuda_error<true>(err, __FILE__, __LINE__);                                            \
	}

/**
 * A function to catch and display errors relayed from CUDA function. 
 * @ingroup debug 
 * @param err Cuda error returned. 
 * @param filename file containing the error
 * @param line line where the error occured
 */
template<bool shouldThrow = false>
void check_cuda_error(const cudaError_t err, const char *filename, const int line)
{
	if(err != cudaSuccess)
	{
		std::cout << "(" << err << ") in " << filename << " on line " << line << " : "
		          << cudaGetErrorString(err) << std::endl;
		if constexpr(shouldThrow)
		{
			throw std::runtime_error(cudaGetErrorString(err));
		}
	}
}

/**
 * A function to catch and display errors relayed from CUDA functions. 
 * @ingroup debug 
 * @param code the error code returned from a wrapped CUDA function. 
 * @param filename file containing the error
 * @param line line where the error occured
 */
template<bool shouldThrow = false>
void check_cuda_error(const CUresult code, const char *filename, const int line)
{
	if(code != CUDA_SUCCESS)
	{
		const char *name = nullptr;
		const char *strg = nullptr;

		cuGetErrorName(code, &name);
		cuGetErrorString(code, &strg);

		std::cout << "(" << code << ") in " << filename << " on line " << line << " : " << std::string(name)
		          << " - " << std::string(strg) << std::endl;
		if constexpr(shouldThrow)
		{
			throw std::runtime_error("CU CUDA Error");
		}
	}
}
#endif