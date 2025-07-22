#include <mpi.h>
#include <iostream>


#include "helpers.hpp"
#ifdef USE_CXI
#include "safety/hip.hpp"
#endif

extern "C" {
#ifdef USE_CXI
int MPI_Init_thread(int* argc, char*** argv, int required, int* provided)
{
    force_hip(hipInit(0));
    force_hip(hipSetDevice(6));
    int error_code = PMPI_Init_thread(argc, argv, required, provided);
    init_debugs();
    return error_code;
}
#endif

int MPI_Init(int* argc, char*** argv)
{
    force_hip(hipInit(0));
    force_hip(hipSetDevice(6));
    int error_code = PMPI_Init(argc, argv);
    init_debugs();
    return error_code;
}

void MPIS_Hello_world()
{
    std::cout << "Hello world!" << std::endl;
}
}
