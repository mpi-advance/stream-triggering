#ifndef ST_MEM_POOL
#define ST_MEM_POOL

#include <cstring>
#include <memory_resource>
#include <new>
#include <utility>

#include "safety/gpu.hpp"

/**
 * @file mem_pool.hpp
 * @brief Custom memory resources for host and GPU memory allocated from GPU library
 * calls.
 * @details Technically no pools of memory at the moment.
 */

/** @brief The collection of all memory-related helper functions */
namespace Memory
{

/** @brief A modern C++ class that can provide host memory allocated from a GPU library.
 * @details Does not support building with both AMD and NVIDIA GPU libraries at the same
 * time (code will compile, but always default to HIP). If no GPU libraries are enabled,
 * the code will compile, but all allocations will throw std::bad_alloc.
 */
class gpu_host_memory_resource : public std::pmr::memory_resource
{
protected:
    /** @brief Allocate zeroed out CPU memory from the GPU library call.
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

    /** @brief Free the memory allocated by gpu_host_memory_resource::do_allocate()
     * @details While this method has other parameters to it (bytes, alignment),
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

    /** @brief Check whether memory from this resource can be freed by another resource.
     * @details Two Memory::gpu_host_memory_resource objects are equal only if they are
     * the same object. This is the strictest possible definition, so std::pmr containers
     * will never try to free memory through a resource that didn't allocate it.
     * @param [in] other The memory resource to compare against.
     * @return True if @p other is this exact object, false otherwise.
     */
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
    {
        return this == &other;
    }
};

/** @brief Singleton memory resource for GPU Host memory.
 * @return Pointer to the singleton memory pool (Memory::gpu_host_memory_resource) for GPU
 * host memory allocations.
 */
inline gpu_host_memory_resource* gpu_host_resource()
{
    static gpu_host_memory_resource the_resource;
    return &the_resource;
}

/** @brief A modern C++ class that can provide GPU memory allocated from a GPU library.
 * @details Does not support building with both AMD and NVIDIA GPU libraries at the same
 * time (code will compile, but always default to HIP). If no GPU libraries are enabled,
 * the code will compile, but all allocations will throw std::bad_alloc.
 * Code current calls hipMalloc or cudaMalloc for allocation.
 */
class gpu_device_memory_resource : public std::pmr::memory_resource
{
protected:
    /** @brief Allocate GPU memory from the GPU library call.
     * @details The memory is left uninitialized; it defaults to whatever the underlying
     * GPU library does. The alignment parameter is ignored since the GPU allocation
     * functions seem to handle this on their own (and do not take an alignment value).
     * @param [in] bytes How much memory to allocate (in bytes).
     * @return A pointer to the newly allocated memory of size "bytes".
     */
    void* do_allocate(std::size_t bytes, std::size_t) override
    {
        void* ptr = nullptr;
#ifdef HIP_GPUS
        force_gpu(hipMalloc(&ptr, bytes));
#elif defined(CUDA_GPUS)
        force_gpu(cudaMalloc(&ptr, bytes));
#endif
        if (nullptr == ptr)
        {
            throw std::bad_alloc{};
        }

        return ptr;
    }

    /** @brief Free the memory allocated by gpu_device_memory_resource::do_allocate()
     * @details While this method has other parameters to it (bytes, alignment),
     * since the GPU library calls do not care about them, they are ignored in this
     * function. The exact freeing function depends on which GPU library is enabled.
     * @param [in] ptr The memory allocation to free.
     */
    void do_deallocate(void* ptr, std::size_t, std::size_t) override
    {
#ifdef HIP_GPUS
        check_gpu(hipFree(ptr));
#elif defined(CUDA_GPUS)
        check_gpu(cudaFree(ptr));
#endif
    }

    /** @brief Check whether memory from this resource can be freed by another resource.
     * @details Two Memory::gpu_device_memory_resource objects are equal only if they are
     * the same object. This is the strictest possible definition, so std::pmr containers
     * will never try to free memory through a resource that didn't allocate it.
     * @param [in] other The memory resource to compare against.
     * @return True if @p other is this exact object, false otherwise.
     */
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
    {
        return this == &other;
    }
};

/** @brief Singleton memory resource for GPU device memory.
 * @return Pointer to the singleton memory pool (Memory::gpu_device_memory_resource) for
 * GPU memory allocations.
 */
inline gpu_device_memory_resource* gpu_device_resource()
{
    static gpu_device_memory_resource the_resource;
    return &the_resource;
}

/** @brief Wrapper around a buffer allocation that will deallocate upon destruction.
 * @details Uses a C++ memory resource (such as Memory::gpu_host_resource() or
 * Memory::gpu_device_resource()) to allocate and deallocate the buffer. Class cannot be
 * copied, but can be move-constructed (leaves the original with nullptr afterwards).
 * The goal of this class was to more easily control when a buffer is free when it sits
 * inside containers of other objects. This is helpful when the buffer may be tied to
 * other resources, such as libfabric memory regions as seen in the CXIQueue class.
 *
 * The memory resource provided to this class for allocation and deallocation must outlive
 * Memory::OwnedBuffer it is provided to.
 */
class OwnedBuffer
{
public:
    /** @brief Create a new buffer of size @p _size bytes from the provided memory
     * resource.
     * @param [in] _resource A pointer to the C++ std::pmr::memory_resource to use.
     * @param [in] _size The number of bytes to allocate for the buffer.
     */
    OwnedBuffer(std::pmr::memory_resource* _resource, std::size_t _size)
        : resource(_resource), size(_size), ptr(_resource->allocate(_size))
    {
    }

    /** @brief Destructor that frees the buffer if it is not nullptr. */
    ~OwnedBuffer()
    {
        if (nullptr != ptr)
        {
            resource->deallocate(ptr, size);
        }
    }

    /** @brief Deleted copy constructor. */
    OwnedBuffer(const OwnedBuffer&) = delete;
    /** @brief Deleted copy assignment operator. */
    OwnedBuffer& operator=(const OwnedBuffer&) = delete;

    /** @brief Move constructor.
     * @details Copies OwnedBuffer::resource and OwnedBuffer::size, but moves
     * OwnedBuffer::ptr and sets the value in @p other to nullptr.
     * @param [in, out] other The OwnedBuffer object to move from.
     */
    OwnedBuffer(OwnedBuffer&& other) noexcept
        : resource(other.resource),
          size(other.size),
          ptr(std::exchange(other.ptr, nullptr))
    {
    }

    /** @brief Deleted move assignment operator. */
    OwnedBuffer& operator=(OwnedBuffer&&) = delete;

    /** @brief Access the underlying raw pointer.
     * @return The underlying raw pointer. Will be nullptr after a move.
     */
    void* get() const
    {
        return ptr;
    }

private:
    /** @brief Memory resource that was allocated from (and to deallocate back into). */
    std::pmr::memory_resource* resource;
    /** @brief The size of the underlying buffer, in bytes. */
    std::size_t size;
    /** @brief The raw pointer of the buffer represented by this class. */
    void* ptr;
};

}  // namespace Memory

#endif