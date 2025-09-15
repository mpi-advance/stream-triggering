#include "helpers.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_GPU_mem_alloc(MPI_Aint size, MPI_Info info, void** baseptr)
{
    std::function<void()> delete_fn;
    if (MPI_INFO_NULL != info)
    {
        constexpr int string_size = 10;
        char          info_key[]  = "MPIS_GPU_MEM_TYPE";
        char          value[string_size];
        int           flag = 0;
        // Pre MPI-4.0
        force_mpi(MPI_Info_get(info, info_key, string_size, value, &flag));
        if (!flag)
        {
            throw std::runtime_error("Required Info Key Missing: MPIS_GPU_MEM_TYPE.");
        }

        if (0 == strcmp(value, "FINE"))
        {
#if defined(HIP_GPUS)
            force_gpu(hipExtMallocWithFlags(baseptr, size, hipDeviceMallocFinegrained));
            delete_fn = [baseptr]() { check_gpu(hipFree(*baseptr)); };
#else
            throw std::runtime_error(
                "Library was not built with required GPU support enabled.");
#endif
        }
        else if (0 == strcmp(value, "COARSE"))
        {
#ifdef CUDA_GPUS
            force_gpu(cudaMalloc(baseptr, size));
            delete_fn = [baseptr]() { check_gpu(cudaFree(*baseptr)); };
#elif defined(HIP_GPUS)
            force_gpu(hipMalloc(baseptr, size));
            delete_fn = [baseptr]() { check_gpu(hipFree(*baseptr)); };
#else
            throw std::runtime_error(
                "Library was not built with required GPU support enabled.");
#endif
        }
        /* Also offer a case for host memory, but not sure if this is useful (especially
         * since no GPU tie in.)*/
        else if (0 == strcmp(value, "HOST"))
        {
            *baseptr  = operator new(size);
            delete_fn = [baseptr]() { operator delete(*baseptr); };
        }
        else
        {
            throw std::runtime_error("Unsupported value for MPIS_GPU_MEM_TYPE.");
        }
    }
    else
    {
        throw std::runtime_error("Required Info Key Missing: MPIS_GPU_MEM_TYPE.");
    }

    /* Register pointer and it's freeing function. */
    deletors[*baseptr] = delete_fn;

    return MPIS_SUCCESS;
}
}