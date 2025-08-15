#include "abstract/queue.hpp"
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
#ifdef USE_HPE
#include <hip/hip_runtime.h>
#include "queues/HPEQueue.hpp"
#endif
#ifdef USE_CXI
#include <hip/hip_runtime.h>
#include "queues/CXIQueue.hpp"
#endif

#include <stdexcept>

extern "C" {

int MPIS_Queue_init(MPIS_Queue *queue, MPIS_Queue_type type, void* extra_address)
{
	Queue *the_queue;
	switch(type)
	{
#ifdef USE_THREADS
		case THREAD:
			the_queue = new ThreadQueue<false>();
			break;
		case THREAD_SERIALIZED:
			the_queue = new ThreadQueue<true>();
			break;
#endif
#ifdef USE_MEM_OPS
		case GPU_MEM_OPS:
	#ifdef CUDA_GPUS
			the_queue = new CudaQueue((cudaStream_t *) (extra_address));
			break;
	#endif
	#ifdef HIP_GPUS
			the_queue = new HIPQueue((hipStream_t*) (extra_address));
			break;
	#endif
#endif
#ifdef USE_HPE
		case HPE:
			the_queue = new HPEQueue((hipStream_t *) (extra_address));
			break;
#endif
#ifdef USE_CXI
		case CXI:
			the_queue = new CXIQueue((hipStream_t *) (extra_address));
			break;
#endif
		default:
			throw std::runtime_error("Queue type not enabled");
	}
	*queue = (MPIS_Queue) the_queue;
    
	if(MPIS_QUEUE_NULL == ACTIVE_QUEUE)
	{
		ACTIVE_QUEUE = (MPIS_Queue) the_queue;
	}
	else
	{
		throw std::runtime_error("There is already an active queue");
	}

	return MPIS_SUCCESS;
}
}