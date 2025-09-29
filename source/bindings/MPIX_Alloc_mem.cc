#include "helpers.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Alloc_mem(MPI_Aint size, MPI_Info info, void** baseptr)
{
    std::function<void()> delete_fn;
    if (MPI_INFO_NULL != info)
    {
        constexpr int string_size = 100;
        char          info_key[]  = "mpi_memory_alloc_kinds";
        char          value[string_size];
        int           flag = 0;
        // Pre MPI-4.0
        force_mpi(MPI_Info_get(info, info_key, string_size, value, &flag));
        if (!flag)
        {
            throw std::runtime_error("Required Info Key Missing: mpi_memory_alloc_kinds");
        }

        Print::out("Desired Memory Type:", value);

        if (0 == strcmp(value, "rocm:device:fine"))
        {
#if defined(HIP_GPUS)
            Print::out("Created:", "rocm:device:fine");
            force_gpu(hipExtMallocWithFlags(baseptr, size, hipDeviceMallocFinegrained));
            auto p_val = *baseptr;
            delete_fn = [p_val]() { check_gpu(hipFree(p_val)); };
#else
            throw std::runtime_error(
                "Library was not built with required GPU support enabled.");
#endif
        }
        else if (0 == strcmp(value, "rocm:device:coarse") ||
                 0 == strcmp(value, "rocm:device"))
        {
#if defined(HIP_GPUS)
            Print::out("Created:", "rocm:device");
            force_gpu(hipMalloc(baseptr, size));
            auto p_val = *baseptr;
            delete_fn = [p_val]() { check_gpu(hipFree(p_val)); };
#else
            throw std::runtime_error(
                "Library was not built with required GPU support enabled.");
#endif
        }
        else if (0 == strcmp(value, "cuda:device"))
        {
#if defined(CUDA_GPUS)
            Print::out("Created:", "cuda:device");
            force_gpu(cudaMalloc(baseptr, size));
            auto p_val = *baseptr;
            delete_fn = [p_val]() { check_gpu(cudaFree(p_val)); };
#endif
        }
        /* Also offer a case for host memory, but not sure if this is useful (especially
         * since no GPU tie in.)*/
        else if (0 == strcmp(value, "system"))
        {
            Print::out("Created:", "system");
            *baseptr  = operator new(size);
            auto p_val = *baseptr;
            delete_fn = [p_val]() { operator delete(p_val); };
        }
        else
        {
            throw std::runtime_error("Unsupported value for: mpi_memory_alloc_kinds");
        }
    }
    else
    {
        throw std::runtime_error("Required Info Key Missing: mpi_memory_alloc_kinds");
    }

    /* Register pointer and it's freeing function. */
    deletors[*baseptr] = delete_fn;

    return MPIS_SUCCESS;
}
}