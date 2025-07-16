#ifndef ST_BINDING_HELPERS
#define ST_BINDING_HELPERS

#include <memory>
#include <type_traits>

#include "abstract/request.hpp"
#include "stream-triggering.h"

#include "misc/print.hpp"

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

#endif