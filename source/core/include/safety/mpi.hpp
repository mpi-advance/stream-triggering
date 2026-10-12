#ifndef ST_MPI_SAFETY
#define ST_MPI_SAFETY

#include <iostream>

#include "mpi.h"

/**
 * @file mpi.hpp
 * @brief Helper macros and functions for ensuring proper error checking on MPI
 * functions
 */

/** @brief Call function and print out if error code is not success.*/
#define check_mpi(function)                       \
    {                                             \
        int err = function;                       \
        check_mpi_error(err, __FILE__, __LINE__); \
    }

/** @brief Call function and raise exception if error code is not success. */
#define force_mpi(function)                             \
    {                                                   \
        int err = function;                             \
        check_mpi_error<true>(err, __FILE__, __LINE__); \
    }

/**
 * @brief Check the return code of a given MPI function for a non-MPI_SUCCESS value.
 * @details This function does not print out any strings related to the error code. Given
 * that the application is likely using MPI_ERRORS_ABORT (the default mode), the MPI
 * implementation will likely print out more helpful information. Regardless, the value of
 * the error code is still printed out.
 *
 * @ingroup safety
 * @tparam shouldThrow If true, throw on non-MPI_SUCCESS code instead of just printing
 * out.
 * @param [in] err The error code returned by the MPI function
 * @param [in] filename The name of the file where the function was called.
 * @param [in] line The line number containing the code
 */
template <bool shouldThrow = false>
void check_mpi_error(const int err, const char* filename, const int line)
{
    if (err != MPI_SUCCESS)
    {
        std::cout << "(" << err << ") in " << filename << " on line " << line
                  << std::endl;
        if constexpr (shouldThrow)
        {
            throw std::runtime_error(std::to_string(err));
        }
    }
}
#endif