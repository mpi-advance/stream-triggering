#ifndef ST_BINDING_HELPERS
#define ST_BINDING_HELPERS

#include <memory>
#include <type_traits>

#include "abstract/request.hpp"
#include "misc/print.hpp"
#include "stream-triggering.h"

#ifdef USE_CXI
#include "safety/hip.hpp"
#endif

enum RequestState
{
    ONGOING   = -1,
    UNMATCHED = 0,
    MATCHED   = 1,
};

struct MPIS_Request_struct
{
    RequestState state;
    uintptr_t    internal_request;
};

struct MPISException : public std::runtime_error
{
    MPISException(int error_code, std::string err_message)
        : runtime_error(err_message), code(error_code)
    {
    }

    int code;
};

static inline void print_device_info()
{
    int device = -1;
    int count = -1;
    force_hip(hipGetDevice(&device));
    force_hip(hipGetDeviceCount(&count));
    Print::out("Current Device:", device, count);

    for(int i = 0; i < count; i++)
    {
        int pci_bus_id = -1;
        int pci_device_id = -1;
        int pci_domain_id = -1;
        force_hip(hipDeviceGetAttribute(&pci_bus_id, hipDeviceAttributePciBusId, i)); 
        force_hip(hipDeviceGetAttribute(&pci_device_id, hipDeviceAttributePciDeviceId, i)); 
        force_hip(hipDeviceGetAttribute(&pci_domain_id, hipDeviceAttributePciDomainID, i));
        Print::out("Others:", i, pci_bus_id, pci_device_id, pci_domain_id);
    }    

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

// Functions for extracting C++ request from C type (if it's correct request
// type)
static inline std::shared_ptr<Communication::Request>* convert_request(
    MPIS_Request request, RequestState state)
{
    if (state != request->state)
    {
        throw MPISException(MPIS_INVALID_REQUEST_STATE, "Invalid Request state!");
    }
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        (request->internal_request));
}

static inline std::shared_ptr<Communication::Request>* convert_request_ptr(
    MPIS_Request* request, RequestState state)
{
    if (state != (*request)->state)
    {
        throw MPISException(MPIS_INVALID_REQUEST_STATE, "Invalid Request state!");
    }
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        ((*request)->internal_request));
}

static inline std::shared_ptr<Communication::Request>* convert_request(
    MPIS_Request request)
{
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        (request->internal_request));
}

static inline std::shared_ptr<Communication::Request>* convert_request_ptr(
    MPIS_Request* request)
{
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        ((*request)->internal_request));
}

#endif