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

/** \todo should we have an include in the bindings folder, just so we are consistent with header placement */
/**
 * @brief Valid operational states for a MPIS_Request
*/
enum RequestState
{
    ONGOING   = -1, //!<request is still being processed. 
    UNMATCHED = 0,  //!<request is waiting for match in queue
    MATCHED   = 1,  //!<request has been matched and completed
};

/**
 * @brief wrapper for internal request with control state
*/
struct MPIS_Request_struct
{
    RequestState state; //!<Current state of the Request
    uintptr_t    internal_request; //!< pointer to internal request, 
	                                
};

/**
 * @brief Error structure to capture if a MPIS_Request is in an invalid state. 
*/
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

/**
  * \defgroup extractor Request Extractors
  * @brief Functions for extracting a pointer to the internal request from a MPIS_Request
  * @details
  *
  *
*/

/** 
 * @brief returns a shared pointer to the internal mpi_request, and checks its status
 * @details 
 * This function checks the state of the request before attempting 
 * to extract a pointer to the internal request.
 * @ingroup extractor
 * @param [in] request The MPIS_Request to be unwrapped
 * @param [in] state The value to check the state of the request against. 
 * @return a shared Communication::Request pointer to the internal request
*/
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

/**
 * @brief returns a shared pointer the internal mpi_request and checks its status
 * @details 
 * This function checks the state of the request before attempting 
 * to extract a pointer to the internal request.
 * @ingroup extractor
 * @param [in] request pointer to the MPIS_Request to unwrap
 * @param [in] state The value to check the state of the request against. 
 * @return a shared Communication::Request pointer to the internal request
*/
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

/**
/** 
 * @brief returns a shared pointer to the internal mpi_request
 * @details 
 * This function checks the state of the request before attempting 
 * to extract a pointer to the internal request.
 * @ingroup extractor
 * @param [in] request The MPIS_Request to be unwrapped
 * @return a shared Communication::Request pointer to the internal request
*/
static inline std::shared_ptr<Communication::Request>* convert_request(
    MPIS_Request request)
{
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        (request->internal_request));
}

/**
 * @brief returns a shared pointer the internal mpi_request
 * @details 
 * This function checks the state of the request before attempting 
 * to extract a pointer to the internal request.
 * @ingroup extractor
 * @param [in] request pointer to the MPIS_Request to unwrap
 * @return a shared Communication::Request pointer to the internal request
*/
static inline std::shared_ptr<Communication::Request>* convert_request_ptr(
    MPIS_Request* request)
{
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        ((*request)->internal_request));
}

#endif