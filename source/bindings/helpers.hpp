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

static inline void init_device()
{
#ifdef USE_GFX90A
    force_hip(hipInit(0));
    int device = -1;
    int count = -1;
    force_hip(hipGetDevice(&device));
    force_hip(hipGetDeviceCount(&count));
    std::cout << "info: " << device << " " << count << std::endl;
    force_hip(hipSetDevice(6));
    Print::out("Initialized Device to 6");
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