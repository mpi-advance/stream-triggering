#include <mpi.h>

#include "helpers.hpp"

extern "C" {

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided)
{
    MPIS_BINDING_ENTER
    int error_code = PMPI_Init_thread(argc, argv, required, provided);
    initialize_st();
    MPIS_BINDING_EXIT
    return error_code;
}

int MPI_Init(int* argc, char*** argv)
{
    MPIS_BINDING_ENTER
    int error_code = PMPI_Init(argc, argv);
    initialize_st();
    MPIS_BINDING_EXIT
    return error_code;
}

int MPI_Finalize()
{
    MPIS_BINDING_ENTER
    check_mpi(MPI_Comm_free(&Communication::MPIS_COMM_WORLD));
    int error_code = PMPI_Finalize();
    MPIS_BINDING_EXIT
    return error_code;
}
}
