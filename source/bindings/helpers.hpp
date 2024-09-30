#ifndef ST_BINDING_HELPERS
#define ST_BINDING_HELPERS

#include <memory>
#include <type_traits>

#include "abstract/request.hpp"
#include "stream-triggering.h"

// Functions for converting to C++ Type
template <typename T>
concept MPIS_Type = std::is_same_v<T, uintptr_t>;

template <typename STType, MPIS_Type MPISType>
static inline std::shared_ptr<STType> convert_from_mpis(MPISType mpi_obj)
{
    return *reinterpret_cast<std::shared_ptr<STType>*>(mpi_obj);
}

template <typename STType, MPIS_Type MPISType>
static inline std::shared_ptr<STType>* convert_from_mpis_ptr(MPISType* mpi_obj)
{
    return reinterpret_cast<std::shared_ptr<STType>*>((*mpi_obj));
}

static inline std::shared_ptr<Communication::Request> convert_request(
    MPIS_Request request)
{
    return convert_from_mpis<Communication::Request, MPIS_Request>(request);
}

static inline std::shared_ptr<Communication::Request>* convert_request(
    MPIS_Request* request)
{
    return convert_from_mpis_ptr<Communication::Request, MPIS_Request>(request);
}

#endif