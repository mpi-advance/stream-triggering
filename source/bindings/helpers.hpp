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
/**
 * Valid operational states for a MPIS_Request
*/
enum RequestState
{
    ONGOING   = -1, ///<request is still being processed. 
    UNMATCHED = 0, ///<request is waiting for match in queue
    MATCHED   = 1, ///<request has been matched and completed
};

/**
 * wrapper for internal request with control state
*/
struct MPIS_Request_struct
{
    RequestState state; ///<Current state of the Request
    uintptr_t    internal_request; ///< pointer to internal request, 
	                               ///< will be cast as necessary. 
};

/**
 * Error structure, to capture if a MPIS_Request is in an invalid state. 
*/
struct MPISException : public std::runtime_error
{
    MPISException(int error_code, std::string err_message)
        : runtime_error(err_message), code(error_code)
    {
    }

    int code;
};


/**
* Function to provide additional information for debugging
MPI_init. 
**/
static inline void init_debugs()
{
    // #ifndef NDEBUG
    //  Setup printing rank
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Print::initialize_rank(rank);
    Print::out("Initialized");
    // #endif
}

/**
* Function to setup HIP device if signaled. 
**/
static inline void init_device()
{
#ifdef USE_GFX90A
    force_hip(hipInit(0));
    force_hip(hipSetDevice(6));
    Print::out("Initialized Device to 6");
#endif
}
/**
  * \defgroup extractor Request Extractors
  * Functions for extracting C++ request from C type (if it's correct request type)
  * @{
*/

/**
 * This function checks the state of the request before attempting 
 * to extract a pointer to the internal request.
 * @param request The MPIS_Request to be unwrapped
 * @param state The value to check the state of the request against. 
 * @return a shared Communication::Request pointer to the internal request of the MPIS_Request
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
 * This function checks the state of the request before attempting 
 * to extract a pointer to the internal request.
 * @param request The MPIS_Request to be unwrapped
 * @param state The value to check the state of the request against. 
 * @return a shared Communication::Request pointer to a pointer to the 
 *         internal request of the MPIS_Request
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
 * This function extracts a pointer to the internal request of the supplied MPIS_Request
 * @param request The MPIS_Request to be unwrapped
 * @return a shared Communication::Request pointer to the 
 *         internal request of the MPIS_Request
*/
static inline std::shared_ptr<Communication::Request>* convert_request(
    MPIS_Request request)
{
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        (request->internal_request));
}

/**
 * This function extracts a pointer to the internal request of the supplied MPIS_Request
 * @param request The MPIS_Request to be unwrapped
 * @return a shared Communication::Request pointer to a pointer to the 
 *         internal request of the MPIS_Request
*/
static inline std::shared_ptr<Communication::Request>* convert_request_ptr(
    MPIS_Request* request)
{
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        ((*request)->internal_request));
}

  /**@}*/
#endif