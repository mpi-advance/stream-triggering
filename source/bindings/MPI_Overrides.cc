#include <mpi.h>

#ifdef USE_CXI
#include "safety/hip.hpp"
extern "C" {
int MPI_Init_thread(int* argc, char*** argv, int required, int* provided)
{
    force_hip(hipInit(0));
    force_hip(hipSetDevice(6));
    return PMPI_Init_thread(argc, argv, required, provided);
}
}
#endif
