#include "abstract/queue.hpp"
#include "helpers.hpp"
#include "misc/print.hpp"
#include "stream-triggering.h"

#ifdef USE_THREADS
#include "queues/ThreadQueue.hpp"
#endif
#ifdef USE_MEM_OPS
#ifdef CUDA_GPUS
#include "cuda.h"
#include "queues/CudaQueue.hpp"
#endif
#ifdef HIP_GPUS
#include "queues/HIPQueue.hpp"
#endif
#endif
#ifdef USE_CXI
#include <hip/hip_runtime.h>

#include "queues/cxi/CXIQueue.hpp"
#endif

#include <stdexcept>

// Picks whichever backend may be compiled in.
// Current ordering is just numerical and has no real reason for it
static Queue* select_default_queue(void* extra_address)
{
#if defined(USE_THREADS)
    return new ThreadQueue();
#elif defined(USE_MEM_OPS) && defined(CUDA_GPUS)
    return new CudaQueue((cudaStream_t*)(extra_address));
#elif defined(USE_MEM_OPS) && defined(HIP_GPUS)
    return new HIPQueue((hipStream_t*)(extra_address));
#elif defined(USE_CXI)
    return new CXIQueue((hipStream_t*)(extra_address));
#else
    throw std::runtime_error("Queue type not enabled");
#endif
}

extern "C" {

int MPIS_Queue_init(MPIS_Queue* queue, MPIS_Queue_type type, void* extra_address)
{
    MPIS_BINDING_ENTER

    Queue* the_queue;
    switch (type)
    {
        case DEFAULT:
            the_queue = select_default_queue(extra_address);
            break;
#ifdef USE_THREADS
        case THREAD:
            the_queue = new ThreadQueue();
            break;
#endif
#ifdef USE_MEM_OPS
        case GPU_MEM_OPS:
#ifdef CUDA_GPUS
            the_queue = new CudaQueue((cudaStream_t*)(extra_address));
            break;
#endif
#ifdef HIP_GPUS
            the_queue = new HIPQueue((hipStream_t*)(extra_address));
            break;
#endif
#endif
#ifdef USE_CXI
        case CXI:
            the_queue = new CXIQueue((hipStream_t*)(extra_address));
            break;
#endif
        default:
            throw std::runtime_error("Queue type not enabled");
    }
    *queue = (MPIS_Queue)the_queue;

    if (MPIS_QUEUE_NULL == ACTIVE_QUEUE)
    {
        ACTIVE_QUEUE = (MPIS_Queue)the_queue;
    }
    else
    {
        throw std::runtime_error("There is already an active queue");
    }

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}