#ifndef ST_BINDING_HELPERS
#define ST_BINDING_HELPERS

#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <source_location>

#include "abstract/request.hpp"
#include "misc/initialize.hpp"
#include "misc/print.hpp"
#include "stream-triggering.h"

#ifdef USE_CXI
#include "safety/gpu.hpp"
#endif

extern MPIS_Queue                             ACTIVE_QUEUE;
extern std::map<void*, std::function<void()>> deletors;

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

#define MPIS_BINDING_ENTER                             \
    auto location = std::source_location::current();   \
    Print::out("Entering:", location.function_name());

#define MPIS_BINDING_EXIT                              \
    Print::out("Exiting:", location.function_name());

#endif