#ifndef ST_MPI_SAFETY
#define ST_MPI_SAFETY

#include "mpi.h"

#include <iostream>

/**
 * wrapper around MPI Functions to note errors, but not stop the 
 * execution so downstream effects can be observed. 
 */
#define check_mpi(function)                                                                        \
	{                                                                                              \
		int err = function;                                                                        \
		check_mpi_error(err, __FILE__, __LINE__);                                                  \
	}

/**
 * wrapper around MPI Functions to note errors, and terminate execution
 * in a clean method 
 */
#define force_mpi(function)                                                                        \
	{                                                                                              \
		int err = function;                                                                        \
		check_mpi_error<true>(err, __FILE__, __LINE__);                                            \
	}


/**
 * A function to catch and display errors relayed from mpi_functions. 
 * @param err the error code returned from a wrapped MPI function. 
 * @param filename file containing the error
 * @param line line where the error occured
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