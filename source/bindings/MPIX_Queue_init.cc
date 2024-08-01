#include "abstract/queue.hpp"
#include "stream-triggering.h"

#ifdef USE_THREADS
#include "queues/ThreadQueue.hpp"
#endif
#ifdef USE_CUDA
#include "cuda.h"
#include "queues/CudaQueue.hpp"
#endif
#ifdef USE_HPE
#include <hip/hip_runtime.h>
#include "queues/HPEQueue.hpp"
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
#ifdef USE_CUDA
		case CUDA:
			the_queue = new CudaQueue((cudaStream_t *) (extra_address));
			break;
#endif
#ifdef USE_HPE
		case HPE:
			the_queue = new HPEQueue((hipStream_t *) (extra_address));
			break;
#endif
#ifdef USE_LIBFABRIC
		case LIBFABRIC:
			the_queue = new LibfabricQueue();
			break;
#endif
		default:
			throw std::runtime_error("Queue type not enabled");
	}
	*queue = (MPIS_Queue) the_queue;
    
	return MPIS_SUCCESS;
}
}