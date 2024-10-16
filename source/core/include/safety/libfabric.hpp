#ifndef ST_LIBFABRIC_SAFETY
#define ST_LIBFABRIC_SAFETY

#include <rdma/fabric.h>

#include <iostream>

#define check_libfabric(function)             \
    {                                         \
        int err = function;                   \
        check_error(err, __FILE__, __LINE__); \
    }

#define force_libfabric(function)                   \
    {                                               \
        int err = function;                         \
        check_error<true>(err, __FILE__, __LINE__); \
    }

template <bool shouldThrow = false>
void check_error(const int err, const char* filename, const int line)
{
    if (err < 0)
    {
        // fi_strerror gets the string that explains the error
        std::cout << "(" << err << ") in " << filename << " on line " << line
                  << ": " << fi_strerror(-err) << std::endl;
        if constexpr (shouldThrow)
        {
            throw std::runtime_error(fi_strerror(-err));
        }
    }
}

#endif