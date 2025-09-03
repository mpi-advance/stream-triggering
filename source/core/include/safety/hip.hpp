#ifndef ST_HIP_SAFETY
#define ST_HIP_SAFETY

#include <hip/hip_runtime.h>
#include <iostream>

/**
 * wrapper around HIP Functions to note errors, but not stop the 
 * execution so downstream effects can be observed. 
 */
#define check_hip(function)                                                                        \
	{                                                                                              \
		auto err = function;                                                                       \
		check_hip_error(err, __FILE__, __LINE__);                                                  \
	}

/**
 * wrapper around HIP Functions to note errors and cleanly stop execution 
 */
#define force_hip(function)                                                                        \
	{                                                                                              \
		auto err = function;                                                                       \
		check_hip_error<true>(err, __FILE__, __LINE__);                                            \
	}
	
/**
 * A function to catch and display errors relayed from HIP functions. 
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