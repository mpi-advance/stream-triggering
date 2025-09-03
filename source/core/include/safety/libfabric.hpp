#ifndef ST_LIBFABRIC_SAFETY
#define ST_LIBFABRIC_SAFETY

#include <rdma/fabric.h>

#include <iostream>

/**
 * wrapper around CUDA Functions to note errors, but not stop the 
 * execution so downstream effects can be observed. 
 */
#define check_libfabric(function)             \
    {                                         \
        int err = function;                   \
        check_error(err, __FILE__, __LINE__); \
    }

/**
 * wrapper around libfabric Functions to note errors, and stop the 
 * execution 
 */
#define force_libfabric(function)                   \
    {                                               \
        int err = function;                         \
        check_error<true>(err, __FILE__, __LINE__); \
    }

	
/**
 * A function to catch and display errors relayed from HIP functions. 
 * @param err the error code returned from a wrapped libfabric function. 
 * @param filename file containing the error
 * @param line line where the error occured
 */
template <bool shouldThrow = false>
void check_error(const int err, const char* filename, const int line)
{
    if (err < 0)
    {
        // fi_strerror gets the string that explains the error
        std::cout << "(" << err << ") in " << filename << " on line " << line
                  << ": " << fi_strerror(-err) << std::endl;
        if constexpr (shouldThrow)
        {
            throw std::runtime_error(fi_strerror(-err));
        }
    }
}

#endif