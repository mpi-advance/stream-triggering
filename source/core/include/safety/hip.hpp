#ifndef ST_HIP_SAFETY
#define ST_HIP_SAFETY

#include <hip/hip_runtime.h>
#include <iostream>

#define check_hip(function)                                                                        \
	{                                                                                              \
		auto err = function;                                                                       \
		check_hip_error(err, __FILE__, __LINE__);                                                  \
	}

#define force_hip(function)                                                                        \
	{                                                                                              \
		auto err = function;                                                                       \
		check_hip_error<true>(err, __FILE__, __LINE__);                                            \
	}

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