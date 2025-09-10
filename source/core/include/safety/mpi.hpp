#ifndef ST_MPI_SAFETY
#define ST_MPI_SAFETY

#include "mpi.h"

#include <iostream>

/** @brief wrapper around MPI functions to note errors and continue */
#define check_mpi(function)                                                                        \
	{                                                                                              \
		int err = function;                                                                        \
		check_mpi_error(err, __FILE__, __LINE__);                                                  \
	}

/** @brief wrapper around MPI functions to catch errors and terminate */
#define force_mpi(function)                                                                        \
	{                                                                                              \
		int err = function;                                                                        \
		check_mpi_error<true>(err, __FILE__, __LINE__);                                            \
	}


/** @brief A function to catch and display errors relayed from mpi_functions. 
 * @param err the error code returned from a wrapped MPI function. 
 * @param filename file containing the error
 * @param line line where the error occurred
 */
template<bool shouldThrow = false>
void check_mpi_error(const int err, const char *filename, const int line)
{
	if(err != MPI_SUCCESS)
	{
		std::cout << "(" << err << ") in " << filename << " on line " << line << std::endl;
		if constexpr(shouldThrow)
		{
			throw std::runtime_error(std::to_string(err));
		}
	}
}
#endif