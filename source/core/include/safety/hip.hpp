#ifndef ST_HIP_SAFETY
#define ST_HIP_SAFETY

#include <hip/hip_runtime.h>
#include <iostream>

/** @brief wrapper around HIP functions to note errors and continue 
 * @ingroup debug 
 */
#define check_hip(function)                                                                        \
	{                                                                                              \
		auto err = function;                                                                       \
		check_hip_error(err, __FILE__, __LINE__);                                                  \
	}

/** @brief wrapper around HIP functions to catch errors and terminate
 * @ingroup debug 
 */
#define force_hip(function)                                                                        \
	{                                                                                              \
		auto err = function;                                                                       \
		check_hip_error<true>(err, __FILE__, __LINE__);                                            \
	}
	
/**
 * A function to catch and display errors relayed from HIP functions. 
 * @ingroup debug 
 * @param err the error code returned from a wrapped HPI function. 
 * @param filename file containing the error
 * @param line line where the error occured
 */
template<bool shouldThrow = false>
void check_hip_error(const hipError_t err, const char *filename, const int line)
{
	if(err != hipSuccess)
	{
		std::cout << "(" << err << ") in " << filename << " on line " << line << " : "
		          << hipGetErrorString(err) << std::endl;
		if constexpr(shouldThrow)
		{
			throw std::runtime_error(hipGetErrorString(err));
		}
	}
}
#endif