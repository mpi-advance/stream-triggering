#ifndef ST_INIT
#define ST_INIT

#include <cstdlib>

#include "abstract/request.hpp"
#include "print.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

/**
 * @file initialize.hpp
 * @brief File for code related to initializing the stream-triggering library.
 */

/** @brief A diagnostic function that prints out GPU device information
 * @details This function looks over all of the visible devices and prints out the ID, Bus
 * ID, and Domain ID. Does not select a device. Currently only supports doing this for HIP
 * Devices; runs no code when not compiled with HIP.
 */
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

/** @brief Prepare variables that are used when printing debug or error messages.
 * @details Will call Print::initialize_rank to set the rank used in printout messages. If
 * in debug mode, this function will also call ::print_device_info().
 */
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

/** @brief Parse the environment for variables of interest to this library.
 * @details Currently this function checks for the following variables:
 *
 * <table>
 * <tr><th>Environment Variable</th><th>Effect when set</th></tr>
 * <tr><td>`MPIA_ST_DISABLE_IPC`</td>
 * <td>Sets Communication::IPC_PROTOCOL_ENABLED to `false`, which will disable any
 * IPC-based protocols in all Queue classes.
 * </td></tr>
 * <tr><td>`MPIA_ST_DISABLE_CREDIT`</td>
 * <td>Sets Communication::CREDIT_PROTOCOL_ENABLED to `false`, which will disable any
 * Credit-based protocols in all Queue classes.
 * </td></tr>
 * </table>
 *
 * Both flags currently only affect the CXI backend.
 * For all variables checked, a debug printout should be made for quick verification.
 */
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

/** @brief The top level function for initializing the Stream-Triggering library.
 * @details Takes care of setting up various other aspects of the library, along with
 * creating a duplicate of MPI_COMM_WORLD to be used as the "world comm" for this library
 * @sa Communication::MPIS_COMM_WORLD
 */
static inline void initialize_st()
{
    init_debugs();
    init_env();
    force_mpi(MPI_Comm_dup(MPI_COMM_WORLD, &Communication::MPIS_COMM_WORLD));
}

#endif