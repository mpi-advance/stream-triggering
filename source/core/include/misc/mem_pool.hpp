#include <cstring>
#include <memory_resource>

#include "safety/gpu.hpp"

namespace Memory
{

using GPUPool = std::pmr::monotonic_buffer_resource;

class gpu_host_memory_resource : public std::pmr::memory_resource
{
protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override
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

    void do_deallocate(void* ptr, std::size_t, std::size_t) override
    {
#ifdef HIP_GPUS
        check_gpu(hipHostFree(ptr));
#elif defined(CUDA_GPUS)
        check_gpu(cudaFreeHost(ptr));
#endif
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
    {
        return this == &other;
    }
};

}  // namespace Memory