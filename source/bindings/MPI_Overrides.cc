#include <mpi.h>

#include "helpers.hpp"

extern "C" {

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided)
{
    int error_code = PMPI_Init_thread(argc, argv, required, provided);
    init_debugs();
    return error_code;
}

int MPI_Init(int* argc, char*** argv)
{
    int error_code = PMPI_Init(argc, argv);
    init_debugs();
    return error_code;
}

void MPIS_Hello_world()
{
    init_debugs();
}
}
