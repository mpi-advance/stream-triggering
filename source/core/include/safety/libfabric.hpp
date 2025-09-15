#ifndef ST_LIBFABRIC_SAFETY
#define ST_LIBFABRIC_SAFETY

#include <rdma/fabric.h>

#include <iostream>

/** @brief wrapper around libfabric functions to note errors and continue 
 * @ingroup debug 
 */
#define check_libfabric(function)             \
    {                                         \
        int err = function;                   \
        check_error(err, __FILE__, __LINE__); \
    }

/** @brief wrapper around libfabric functions to catch errors and terminate 
 * @ingroup debug 
 */
#define force_libfabric(function)                   \
    {                                               \
        int err = function;                         \
        check_error<true>(err, __FILE__, __LINE__); \
    }

	
/**
 * A function to catch and display errors relayed from HIP functions. 
 * @ingroup debug 
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