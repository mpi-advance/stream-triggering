#ifndef ST_BINDING_HELPERS
#define ST_BINDING_HELPERS

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <source_location>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "abstract/request.hpp"
#include "misc/initialize.hpp"
#include "misc/print.hpp"
#include "stream-triggering.h"

#ifdef USE_CXI
#include "safety/gpu.hpp"
#endif

/**
 * @file helpers.hpp
 * @brief Helper types, globals, and macros shared by the C bindings.
 * @details This file is where all helper objects / types / functions for the bindings are
 * stored. Any code here should never be needed from the library itself, only the
 * bindings.
 */

/** @brief Global variable for a global queue.
 *  @details Technically not needed for semantics purposes, only present
 * for methods whose implementation needs a queue underneath (such as the match methods)
 * The first queue created gets set to this variable, and the variable gets unset when
 * that queue is freed.
 */
extern MPIS_Queue ACTIVE_QUEUE;
/** @brief Global map of all memory allocated by MPIS_Alloc_mem.
 * @details The map contains the address of the memory along with a function that
 * describes how to free the memory. This object is used by MPIS_Free_mem.
 */
extern std::map<void*, std::function<void()>> deletors;

/** @brief The MPIS_Request object given to the user.
 * @details Note that the type of MPIS_Request_struct::internal_request depends on the
 * value of MPIS_Request_struct::match_handle. If not a "match" handle (false), then it is
 * a pointer to a shared pointer of Communication::Request. If a "match" handle, then it
 * is a pointer to a vector of shared pointers to Communication::Request.
 */
struct MPIS_Request_struct
{
    bool      match_handle;  //!< Indication of if this is a "match" request operation.
    uintptr_t internal_request;  //!< Pointer to internal representation of operation.
};

/** @brief Custom exception class that holds an error code. */
struct MPISException : public std::runtime_error
{
    /**
     * @brief Create a stream-triggering exception with a given code and message.
     * @param [in] error_code The error code of the exception.
     * @param [in] err_message The error message of the exception.
     */
    MPISException(int error_code, std::string err_message)
        : runtime_error(err_message), code(error_code)
    {
    }

    int code;  //!< Error code associated with the exception.
};

/**
 * \defgroup converters C <-> C++ Request Conversion Functions
 * @brief Any function that helps convert from the C MPIS_Request to the internal
 * Communicator::Request object.
 * @details More specifically, the functions convert from MPIS_Request (object or pointer)
 * to a shared pointer of Communication::Request (or a pointer to a shared pointer). These
 * functions may also perform state checks on the request to help reduce code duplication.
 */

/** [UPDATE]
 * @brief Convert from MPIS_Request to internal representation. Checks state.
 * @details Converts from MPIS_Request object (not a pointer to one) to a pointer to a
 * shared pointer of Communication::Request. Will also check that the MPIS_Request has the
 * requested ::RequestState before converting.
 *
 * @ingroup converters
 * @param [in] request An MPIS_Request request object to convert from.
 * @param [in] allowed
 * @return Pointer to shared pointer of Communication::Request. Throws exception if
 * invalid state.
 */
static inline std::shared_ptr<Communication::Request>* convert_request(
    MPIS_Request request, std::initializer_list<Communication::Request::State> allowed)
{
    auto* internal = reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        (request->internal_request));
    if (request->match_handle ||
        allowed.end() == std::find(allowed.begin(), allowed.end(), (*internal)->state))
    {
        throw MPISException(MPIS_INVALID_REQUEST_STATE, "Invalid Request state!");
    }
    return internal;
}

/** [UPDATE]
 * @brief Convert from MPIS_Request pointer to internal representation. Checks state.
 * @details Converts from a pointer to an MPIS_Request object to a pointer to a
 * shared pointer of Communication::Request. Will also check that the MPIS_Request has the
 * requested ::RequestState before converting.
 *
 * @ingroup converters
 * @param [in] request An MPIS_Request request object to convert from.
 * @param [in] allowed
 * @return Pointer to shared pointer of Communication::Request. Throws exception if
 * invalid state.
 */
static inline std::shared_ptr<Communication::Request>* convert_request_ptr(
    MPIS_Request* request, std::initializer_list<Communication::Request::State> allowed)
{
    return convert_request(*request, allowed);
}

/**
 * @brief Get internal representation from an MPIS_Request representing a match operation.
 * @details Will throw an MPISException if the MPIS_Request is not from a match operation
 * (aka MPIS_Request::match_handle is false).
 * @param [in] request An MPIS_Request request object to convert from.
 * @return The vector of shared pointers of Communication::Request objects that the input
 * handle was a pointer to.
 */
static inline std::vector<std::shared_ptr<Communication::Request>>* convert_match_request(
    MPIS_Request request)
{
    if (!request->match_handle)
    {
        throw MPISException(MPIS_INVALID_REQUEST_STATE, "Invalid Request state!");
    }
    return reinterpret_cast<std::vector<std::shared_ptr<Communication::Request>>*>(
        (request->internal_request));
}

/**
 * @brief Convert from MPIS_Request to internal representation.
 * @details Converts from MPIS_Request object (not a pointer to one) to a pointer to a
 * shared pointer of Communication::Request. Does not check state of MPIS_Request object.
 *
 * @ingroup converters
 * @param [in] request An MPIS_Request request object to convert from.
 * @return Pointer to shared pointer of Communication::Request.
 */
static inline std::shared_ptr<Communication::Request>* convert_request(
    MPIS_Request request)
{
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        (request->internal_request));
}

/**
 * @brief Convert from MPIS_Request pointer to internal representation.
 * @details Converts from a pointer to an MPIS_Request object to a pointer to a
 * shared pointer of Communication::Request. Does not check state.
 *
 * @ingroup converters
 * @param [in] request An MPIS_Request request object to convert from.
 * @return Pointer to shared pointer of Communication::Request.
 */
static inline std::shared_ptr<Communication::Request>* convert_request_ptr(
    MPIS_Request* request)
{
    return reinterpret_cast<std::shared_ptr<Communication::Request>*>(
        ((*request)->internal_request));
}

/**
 * @brief Helper function that enforces that all Communication::Request in the vector be
 * unique.
 * @details Checks the that all ids (Communication::Request::getID()) are unique by
 * creating a vector of the IDs, sorting it, and then using a C++ algorithm to look for
 * same values. If found, method will throw an exception.
 * @param [in] requests The vector of Communication::Request that must be unique.
 */
static inline void check_unique_requests(
    const std::vector<std::shared_ptr<Communication::Request>>& requests)
{
    std::vector<size_t> ids;
    ids.reserve(requests.size());
    for (const std::shared_ptr<Communication::Request>& req : requests)
    {
        ids.push_back(req->getID());
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
    {
        throw std::runtime_error("The same request was named more than once");
    }
}

/** @brief Helper macro that current only prints out the function name when in debug mode.
 * @details Designed to be called at the start of any stream-triggered C binding.
 */
#define MPIS_BINDING_ENTER                           \
    auto location = std::source_location::current(); \
    Print::out("Entering:", location.function_name());

/** @brief Helper macro that current only prints out the function name when in debug mode.
 * @details Designed to be called before a return in any stream-triggered C binding. The
 * implementation of this macro requires MPIS_BINDING_ENTER to be present in the same
 * scope as where this macro is called.
 */
#define MPIS_BINDING_EXIT Print::out("Exiting:", location.function_name());

#endif