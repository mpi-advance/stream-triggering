#ifndef ST_MEM_POOL
#define ST_MEM_POOL

#include <cstring>
#include <memory_resource>

#include "safety/gpu.hpp"

/**
 * @file mem_pool.hpp
 * @brief Custom memory pool for host memory allocated from GPU library calls.
 */

/** @brief The collection of all memory-related helper functions */
namespace Memory
{

/** @brief Convenience wrapper for long C++ type; the object to allocate host memory from
 */
using GPUPool = std::pmr::monotonic_buffer_resource;

/**
 * @brief A modern C++ class that can provide host memory allocated from a GPU library.
 * @details Does not support building with both AMD and NVIDIA GPU libraries at the same
 * time (code will compile, but always default to HIP). If no GPU libraries are enabled,
 * the code will compile, but all allocations will throw std::bad_alloc.
 */
class gpu_host_memory_resource : public std::pmr::memory_resource
{
protected:
    /**
     * @brief Allocate zeroed out CPU memory from the GPU library call.
     * @details Allocating CPU memory this way often allows the GPU library to pin it and
     * make the GPU aware of its presence (on systems where the GPU can read mapped CPU
     * memory). If the allocation fails (and is somehow not detected by the GPU function
     * macros), or no GPU libraries are enabled, std::bad_alloc is raised. The alignment
     * parameter is ignored since the GPU allocation functions seem to handle this on
     * their own (and do not take an alignment value).
     *
     * Additionally, the memory is also set to 0 with std::memset.
     * @param [in] bytes How much memory to allocate (in bytes).
     * @return A pointer to the newly allocated memory of size "bytes".
     */
    void* do_allocate(std::size_t bytes, std::size_t) override
    {
        void* ptr = nullptr;
#ifdef HIP_GPUS
        force_gpu(hipHostMalloc(&ptr, bytes, hipHostAllocDefault));
#elif defined(CUDA_GPUS)
        force_gpu(cudaMallocHost(&ptr, bytes, cudaHostAllocDefault));
#endif
        if (nullptr == ptr)
        {
            throw std::bad_alloc{};
        }

        std::memset(ptr, 0, bytes);

        return ptr;
    }

    /**
     * @brief Free the memory allocated by ::do_allocate()
     * @details While this method often has other parameters to it (bytes, alignment),
     * since the GPU library calls do not care about them, they are ignored in this
     * function. The exact freeing function depends on which GPU library is enabled.
     * @param [in] ptr The memory allocation to free.
     */
    void do_deallocate(void* ptr, std::size_t, std::size_t) override
    {
#ifdef HIP_GPUS
        check_gpu(hipHostFree(ptr));
#elif defined(CUDA_GPUS)
        check_gpu(cudaFreeHost(ptr));
#endif
    }

    /**
     * @brief Check whether memory from this resource can be freed by another resource.
     * @details Two gpu_host_memory_resource objects are equal only if they are the same
     * object. This is the strictest possible definition, so std::pmr containers will
     * never try to free memory through a resource that didn't allocate it.
     * @param [in] other The memory resource to compare against.
     * @return `true` if `other` is this exact object, `false` otherwise.
     */
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
    {
        return this == &other;
    }
};

}  // namespace Memory

#endif