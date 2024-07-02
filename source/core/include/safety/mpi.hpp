#ifndef ST_MPI_SAFETY
#define ST_MPI_SAFETY

#include "mpi.h"

#include <iostream>

#define check_mpi(function)                                                                        \
	{                                                                                              \
		int err = function;                                                                        \
		check_mpi_error(err, __FILE__, __LINE__);                                                  \
	}

#define force_mpi(function)                                                                        \
	{                                                                                              \
		int err = function;                                                                        \
		check_mpi_error<true>(err, __FILE__, __LINE__);                                            \
	}

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