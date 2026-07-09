#include <mpi.h>

#include "helpers.hpp"

extern "C" {

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided)
{
    int error_code = PMPI_Init_thread(argc, argv, required, provided);
    initialize_st();
    return error_code;
}

int MPI_Init(int* argc, char*** argv)
{
    int error_code = PMPI_Init(argc, argv);
    initialize_st();
    return error_code;
}

void MPIS_Hello_world()
{
    initialize_st();
}
}
