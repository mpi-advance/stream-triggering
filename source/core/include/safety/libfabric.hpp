#ifndef ST_LIBFABRIC_SAFETY
#define ST_LIBFABRIC_SAFETY

#include <rdma/fabric.h>

#include <iostream>

/**
 * @file libfabric.hpp
 * @brief Helper macros and functions for ensuring proper error checking on libfabric
 * functions
 */

/** @brief Call function and print out if error code is not success.*/
#define check_libfabric(function)             \
    {                                         \
        int err = function;                   \
        check_error(err, __FILE__, __LINE__); \
    }

/** @brief Call function and raise exception if error code is not success.*/
#define force_libfabric(function)                   \
    {                                               \
        int err = function;                         \
        check_error<true>(err, __FILE__, __LINE__); \
    }

/**
 * @brief Check the return code of a given libfabric function for a non-zero value.
 * @details If a error code with a value less than zero is given, the function will use
 * fi_strerror to get and print out the error message. Note that some libfabric functions
 * use the return value for dual-purpose, where a positive value (and/or 0) means
 * something to the caller, and a negative value means an error. Further note that some
 * negative "errors" simply mean "not ready" rather than a true impassible error; this
 * function does not consider this context and the caller above should consider that case.
 *
 * @ingroup safety
 * @tparam shouldThrow If true, throw on non-zero error code instead of just printing out.
 * @param [in] err The error code returned by the libfabric function
 * @param [in] filename The name of the file where the function was called.
 * @param [in] line The line number containing the code
 */
template <bool shouldThrow = false>
void check_error(const int err, const char* filename, const int line)
{
    if (err < 0)
    {
        // fi_strerror gets the string that explains the error
        std::cout << "(" << err << ") in " << filename << " on line " << line << ": "
                  << fi_strerror(-err) << std::endl;
        if constexpr (shouldThrow)
        {
            throw std::runtime_error(fi_strerror(-err));
        }
    }
}

#endif