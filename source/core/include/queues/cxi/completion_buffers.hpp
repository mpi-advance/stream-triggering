#ifndef ST_CXI_QUEUE_COMPLETION_BUFFER
#define ST_CXI_QUEUE_COMPLETION_BUFFER

#include "queues/cxi/libfabric_wrappers.hpp"

class CompletionBuffer
{
public:
    CompletionBuffer(void* _address, size_t _len, size_t _count, uint64_t _mr_key,
                     uint64_t _offset)
        : address(_address),
          iov_description({_offset, _len, _mr_key}),
          ioc_description({_offset, _count, _mr_key})
    {
    }

    CompletionBuffer()
        : address(nullptr), iov_description({0, 0, 0}), ioc_description({0, 0, 0})
    {
    }

    operator fi_rma_iov() const
    {
        return iov_description;
    }

    operator fi_rma_ioc() const
    {
        return ioc_description;
    }

    struct fi_rma_iov* get_rma_iov_addr()
    {
        return &iov_description;
    }

    struct fi_rma_ioc* get_rma_ioc_addr()
    {
        return &ioc_description;
    }

    void print()
    {
        Print::out("CB:", address, iov_description.addr, iov_description.len,
                   iov_description.key, "/", ioc_description.addr, ioc_description.count,
                   ioc_description.key);
    }

    void*             address;
    struct fi_rma_iov iov_description;
    struct fi_rma_ioc ioc_description;
};

class CompletionBufferFactory
{
public:
    CompletionBufferFactory() : my_mr(nullptr), current_index(0)
    {
        initialize_main_buffer();
    }

    ~CompletionBufferFactory()
    {
        if (my_mr)
        {
            force_libfabric(fi_close(&(my_mr)->fid));
        }
        check_gpu(hipFree(buffer));
    }

    CompletionBufferFactory(const CompletionBufferFactory& other) = delete;
    CompletionBufferFactory(CompletionBufferFactory&& other)
    {
        buffer       = other.buffer;
        my_mr        = other.my_mr;
        other.buffer = nullptr;
        other.my_mr  = nullptr;
    }

    CompletionBufferFactory& operator=(const CompletionBufferFactory& rhs) = delete;
    CompletionBufferFactory& operator=(CompletionBufferFactory&& other)
    {
        buffer       = other.buffer;
        my_mr        = other.my_mr;
        other.buffer = nullptr;
        other.my_mr  = nullptr;
        return *this;
    }

    hipIpcMemHandle_t get_ipc_handle()
    {
        return ipc_handle;
    }

    void register_mr(LibfabricInstance& libfab)
    {
        my_mr = libfab.create_mr(buffer, DEFAULT_SIZE, FI_REMOTE_WRITE | FI_WRITE,
                                 FI_MR_ALLOCATED);
    }

    void free_mr()
    {
        check_libfabric(fi_close(&(my_mr)->fid));
        my_mr = nullptr;
    }

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

    using COMPLETION_TYPE                       = size_t;
    static constexpr size_t DEFAULT_ITEMS       = 1000;
    static constexpr size_t DEFAULT_ITEM_SIZE   = sizeof(COMPLETION_TYPE);
    static constexpr size_t DEFAULT_SIZE        = DEFAULT_ITEMS * DEFAULT_ITEM_SIZE;
    static constexpr size_t MAX_GPU_COMPLETIONS = 100;

private:
    struct fid_mr*    my_mr;
    void*             buffer;
    size_t            current_index;
    hipIpcMemHandle_t ipc_handle;

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