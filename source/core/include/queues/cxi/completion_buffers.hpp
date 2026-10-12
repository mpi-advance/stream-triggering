#ifndef ST_CXI_QUEUE_COMPLETION_BUFFER
#define ST_CXI_QUEUE_COMPLETION_BUFFER

#include "queues/cxi/libfabric_wrappers.hpp"

/**
 * @file completion_buffers.hpp
 * @brief File for code related to the ::CompletionBuffer abstraction.
 */

/**
 * @brief The class puts together several descriptors to build a "completion buffer"
 * @details A completion buffer is the object that stream-triggered kernels use to figure
 * out if a communication has been completed or not. The exact process rank or mechanism
 * for signalling a process is not decided by this class.
 *
 * Because the library was originally designed around the features of the CXI libfabric
 * provider, the members of this class are in types libfabric often works with. Since it
 * is more of a helper container class, all of the members are public.
 *
 * While this class is the type that CompletionBufferFactory hands out, one does not need
 * to use that class to make its own ::CompletionBuffer. As long as you have the values
 * this class uses, you can use it.
 *
 * As a reminder, the "addr" field in ::CompletionBuffer::iov_description and
 * ::CompletionBuffer::iov_description is an offset value as opposed a real memory
 * address and thus not the same as ::CompletionBuffer::address.
 *
 */
class CompletionBuffer
{
public:
    /**
     * @brief Constructor for a "completion buffer". Saves input variables into various
     * libfabric structs.
     * @details An fi_rma_iov is built from @p _offset, @p _len, and @p _mr_key. An
     * fi_rma_ioc is built from @p _offset, @p _count, and @p _mr_key.
     * Note that the parameters are not saved in any other local members, only these
     * structs (aside from @p _address which does get its own member,
     * CompletionBuffer::address ).
     * @param [in] _address Pointer to the underlying completion buffer.
     * @param [in] _len Size of the completion buffer, in bytes.
     * @param [in] _count The number of elements (usually only 1).
     * @param [in] _mr_key The libfabric memory region key of the buffer.
     * @param [in] _offset Offset of the buffer in its allocation/memory region.
     */
    CompletionBuffer(void* _address, size_t _len, size_t _count, uint64_t _mr_key,
                     uint64_t _offset)
        : address(_address),
          iov_description({_offset, _len, _mr_key}),
          ioc_description({_offset, _count, _mr_key})
    {
    }

    /** @brief Create an empty completion buffer. All values are set to 0 or nullptr. */
    CompletionBuffer()
        : address(nullptr), iov_description({0, 0, 0}), ioc_description({0, 0, 0})
    {
    }

    /** @brief Cast this object into its fi_rma_iov
     * @return The CompletionBuffer::iov_description held by this object
     */
    operator fi_rma_iov() const
    {
        return iov_description;
    }

    /** @brief Cast this object into its fi_rma_ioc
     * @return The CompletionBuffer::ioc_description held by this object
     */
    operator fi_rma_ioc() const
    {
        return ioc_description;
    }

    /** @brief Get the address of the fi_rma_iov member
     * @return The address of CompletionBuffer::iov_description
     */
    struct fi_rma_iov* get_rma_iov_addr()
    {
        return &iov_description;
    }

    /** @brief Get the address of the fi_rma_ioc member
     * @return The address of CompletionBuffer::ioc_description
     */
    struct fi_rma_ioc* get_rma_ioc_addr()
    {
        return &ioc_description;
    }

    /** @brief Helper debug function that prints out the values of all members.*/
    void print()
    {
        Print::out("CB:", address, iov_description.addr, iov_description.len,
                   iov_description.key, "/", ioc_description.addr, ioc_description.count,
                   ioc_description.key);
    }

    /** @brief The completion buffer's memory location */
    void* address;
    /** @brief Description of a completion buffer, in terms of a libfabric fi_rma_iov. */
    struct fi_rma_iov iov_description;
    /** @brief Description of a completion buffer, in terms of a libfabric fi_rma_ioc. */
    struct fi_rma_ioc ioc_description;
};

/**
 * @brief The class designed for handing out pre-allocated ::CompletionBuffers.
 * @details This class allocates a single chunk of GPU (currently HIP) memory, of which it
 * carves out CompletionBufferFactory::DEFAULT_ITEM_SIZE sizds chunks to be used as the
 * location to store GPU-based completion values at. The whole buffer is registered with
 * libfabric to get a memory region key that can be given to peers so that their deferred
 * work queue actions are able to write to a given completion buffer. Because this is a
 * single buffer, many "completion buffers" will have the same libfabric memory region
 * key, but a different offset. This information is all wrapped into the
 * ::CompletionBuffer that is returned from CompletionBufferFactory::alloc_buffer.
 *
 * Buffer allocation happens when the factory is constructed, libfabric registration must
 * manually be invoked later. Allocating before registration will throw an exception.
 *
 * Current implementation of this class only allows it to have 1000
 * (CompletionBufferFactory::DEFAULT_ITEMS) completions, after which the class will throw
 * an exception if more are requested. Slots are not returned when corresponding
 * ::CompletionBuffer objects go away. No technical reason for this other than we have not
 * needed more yet.
 *
 * This class also has a GPU IPC handle available should something need that. The offsets
 * inside the allocated ::CompletionBuffer can be used for IPC- or libfabric-based
 * communications.
 */
class CompletionBufferFactory
{
public:
    /** @brief Create the factory (and underlying buffer).
     * @details Currently only calls CompletionBufferFactory::initialize_main_buffer(),
     * which does have many GPU-based operations besides the allocation. Note that that
     * function currently synchronizes with the GPU device (not stream).
     */
    CompletionBufferFactory() : my_mr(nullptr), current_index(0)
    {
        initialize_main_buffer();
    }

    /** @brief Destructor for class. Close libfabric memory region for buffer (if
     * present); Frees GPU buffer.
     * @details Note that even if the class gets freed, any existing ::CompletionBuffer
     * will not automatically cleaned up until they go out of scope, and should not be
     * used for any communication.
     */
    ~CompletionBufferFactory()
    {
        free_mr();
        check_gpu(hipFree(buffer));
    }

    /** @brief Copy constructor deleted. */
    CompletionBufferFactory(const CompletionBufferFactory& other) = delete;
    /** @brief Move constructor deleted. */
    CompletionBufferFactory(CompletionBufferFactory&& other) = delete;

    /** @brief Copy assignment operator deleted. */
    CompletionBufferFactory& operator=(const CompletionBufferFactory& rhs) = delete;
    /** @brief Move assignment operator deleted. */
    CompletionBufferFactory& operator=(CompletionBufferFactory&& other) = delete;

    /** @brief Access the GPU IPC handle to the underlying buffer.
     * @return The GPU IPC handle of the underlying buffer.
     * @sa CompletionBufferFactory::initialize_main_buffer()
     */
    hipIpcMemHandle_t get_ipc_handle()
    {
        return ipc_handle;
    }

    /**
     * @brief Register the underlying buffer with libfabric
     * @details Will populate CompletionBufferFactory::my_mr with the result. Throws if
     * already registered for a memory_region.
     * @param [in, out] libfab The collection of libfabric objects that can perform the
     * libfabric memory region creation and registration.
     */
    void register_mr(LibfabricInstance& libfab)
    {
        if (my_mr != nullptr)
            throw std::runtime_error("Main buffer already registered with libfabric.");
        my_mr = libfab.create_mr(buffer, DEFAULT_SIZE, FI_REMOTE_WRITE | FI_WRITE,
                                 FI_MR_ALLOCATED);
    }

    /** @brief Deregister the underlying buffer from libfabric.
     * @details Does not free the GPU buffer. Does nothing if the buffer has not been
     * registered yet. Note that calling this with any ::CompletionBuffer objects still
     * being used results in them being invalid for libfabric use.
     */
    void free_mr()
    {
        if (my_mr != nullptr)
        {
            check_libfabric(fi_close(&(my_mr)->fid));
            my_mr = nullptr;
        }
    }

    /** @brief Allocate a buffer for a "completion".
     * @details Allocates a buffer from the current slot in the underlying buffer. If
     * there are no more slots, or the buffer has not been registered with libfabric, an
     * exception is thrown and no buffer is allocated.
     * @return A ::CompletionBuffer with details about the completion buffer.
     */
    CompletionBuffer alloc_buffer()
    {
        if (current_index >= DEFAULT_ITEMS)
            throw std::runtime_error("Out of space for completion buffer");
        if (nullptr == my_mr)
            throw std::runtime_error("Buffer is not registered with libfabric");
        void*    x            = ((char*)buffer) + (sizeof(size_t) * current_index);
        uint64_t offset_value = current_index * DEFAULT_ITEM_SIZE;
        current_index++;
        return CompletionBuffer(x, DEFAULT_ITEM_SIZE, 1, fi_mr_key(my_mr), offset_value);
    }

    /** @brief An alias to be used to describe the type of a "completion". */
    using COMPLETION_TYPE = size_t;
    /** @brief The default number of "completions" the factory will allocate. */
    static constexpr size_t DEFAULT_ITEMS = 1000;
    /** @brief How big each item in the factory is. (size of
     * CompletionBufferFactory::COMPLETION_TYPE). */
    static constexpr size_t DEFAULT_ITEM_SIZE = sizeof(COMPLETION_TYPE);
    /** @brief Simple calculation of the total buffer size in the factory. */
    static constexpr size_t DEFAULT_SIZE = DEFAULT_ITEMS * DEFAULT_ITEM_SIZE;
    /** @brief How many "completions" a ::GPUCompletions object can hold, and how many
     * "triggers" a ::GPUTriggers can hold.*/
    static constexpr size_t MAX_GPU_COMPLETIONS = 100;

private:
    /** @brief The libfabric memory region for the buffer. Needs to be registered before
     * valid. */
    struct fid_mr* my_mr;
    /** @brief The actual buffer from which "completions" are allocated. */
    void* buffer;
    /** @brief Which slot in the buffer to allocate next. */
    size_t current_index;
    /** @brief The GPU (hip) IPC handle for the underlying buffer. */
    hipIpcMemHandle_t ipc_handle;

    /** @brief Helper function that sets up the buffer for the factory.
     * @details Currently allocates the buffer, sets it to zero, synchronizes with the
     * device, and gets an GPU IPC handle for it.
     */
    void initialize_main_buffer()
    {
        force_gpu(hipMalloc(&buffer, DEFAULT_SIZE));
        force_gpu(hipMemset(buffer, 0, DEFAULT_SIZE));
        force_gpu(hipDeviceSynchronize());
        force_gpu(hipIpcGetMemHandle(&ipc_handle, buffer));
        Print::out("Default Completion Buffer location:", buffer);
    }
};

#endif