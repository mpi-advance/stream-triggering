#ifndef ST_CXI_QUEUE_REQUEST_BASE
#define ST_CXI_QUEUE_REQUEST_BASE

#include "abstract/request.hpp"
#include "queues/cxi/completion_buffers.hpp"

enum class TriggerStatus
{
    NOT_NEEDED  = 0,
    GLOBAL_BUMP = 1,
    EXTRA_BUMP  = 2,
    DONE        = 3,  // No wait
};

enum class WaitStatus
{
    NOT_NEEDED    = 0,
    GLOBAL_KERNEL = 1,
};

struct GPUCompletionDescription
{
    volatile CompletionBufferFactory::COMPLETION_TYPE* comp_addr;
    size_t                                             goal_value;
};

struct GPUCompletions
{
    GPUCompletionDescription buffer[CompletionBufferFactory::MAX_GPU_COMPLETIONS];
    size_t                   actual_size;
};

using GPUTriggerDescription = uint64_t*;
struct GPUTriggers
{
    GPUTriggerDescription buffer[CompletionBufferFactory::MAX_GPU_COMPLETIONS];
    size_t                actual_size;
};

class CXIRequest
{
public:
    CXIRequest(Communication::Request& req, CompletionBufferFactory& buffers)
        : base_req(req), completion_buffer(buffers.alloc_buffer()), num_times_started(0)
    {
    }

    // Delayed (or no) buffer setup
    CXIRequest(Communication::Request& req) : base_req(req), num_times_started(0) {}

    virtual ~CXIRequest() = default;
    virtual TriggerStatus start_cpu(CXICounter& trigger_cntr, hipStream_t* the_stream)
    {
        num_times_started++;
        return start_derived(trigger_cntr, the_stream);
    }

    virtual WaitStatus wait_gpu(hipStream_t* the_stream);

    virtual Communication::GPUMemoryType get_gpu_memory_type()
    {
        return base_req.get_memory_type();
    }

    virtual void match(MPI_Comm phase_a, MPI_Comm phase_b) = 0;

    GPUCompletionDescription get_gpu_completion()
    {
        return {(size_t*)completion_buffer.address, num_times_started};
    }

    virtual GPUTriggerDescription get_gpu_trigger()
    {
        throw std::runtime_error("Request was unable to offer trigger description");
    }

protected:
    virtual TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                        hipStream_t* the_stream) = 0;

    Communication::Request& base_req;
    CompletionBuffer        completion_buffer;
    CompletionBuffer        protocol_buffer;

    size_t num_times_started;
};

#endif