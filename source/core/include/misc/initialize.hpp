#ifndef ST_INIT
#define ST_INIT

#include <cstdlib>

#include "abstract/request.hpp"
#include "print.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

static inline void print_device_info()
{
#ifdef HIP_GPUS
    int device = -1;
    int count  = -1;
    force_gpu(hipGetDevice(&device));
    force_gpu(hipGetDeviceCount(&count));
    Print::out("Current Device:", device, count);

    for (int i = 0; i < count; i++)
    {
        int pci_bus_id    = -1;
        int pci_device_id = -1;
        int pci_domain_id = -1;
        force_gpu(hipDeviceGetAttribute(&pci_bus_id, hipDeviceAttributePciBusId, i));
        force_gpu(
            hipDeviceGetAttribute(&pci_device_id, hipDeviceAttributePciDeviceId, i));
        force_gpu(
            hipDeviceGetAttribute(&pci_domain_id, hipDeviceAttributePciDomainID, i));
        Print::out("Others:", i, pci_bus_id, pci_device_id, pci_domain_id);
    }
#endif
}

static inline void init_debugs()
{
    //  Setup printing rank
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Print::initialize_rank(rank);
    Print::out("Initialized");
#ifndef NDEBUG
    print_device_info();
#endif
}

static inline void init_env()
{
    if (std::getenv("MPIA_ST_DISABLE_IPC"))
    {
        Communication::IPC_PROTOCOL_ENABLED = false;
        Print::out("Using IPC:", Communication::IPC_PROTOCOL_ENABLED);
    }
    if (std::getenv("MPIA_ST_DISABLE_CREDIT"))
    {
        Communication::CREDIT_PROTOCOL_ENABLED = false;
        Print::out("Using Credit:", Communication::CREDIT_PROTOCOL_ENABLED);
    }
}

static inline void initialize_st()
{
    init_debugs();
    init_env();
    force_mpi(MPI_Comm_dup(MPI_COMM_WORLD, &Communication::MPIS_COMM_WORLD));
}

#endif