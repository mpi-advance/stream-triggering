#include "abstract/queue.hpp"
#include "stream-triggering.h"

#include <exception>

#ifdef USE_THREADS
#include "queues/ThreadQueue.hpp"
#endif
#ifdef USE_CUDA
#include "cuda.h"
#include "queues/CudaQueue.hpp"
#endif

extern "C" {

int MPIX_Queue_init(MPIX_Queue *queue, MPIX_Queue_type type, void* extra_address)
{
	Queue *the_queue;
	switch(type)
	{
#ifdef USE_CUDA
		case CUDA:
			the_queue = new CudaQueue((cudaStream_t *) (extra_address));
			break;
#endif
#ifdef USE_THREADS
		case THREAD:
			the_queue = new ThreadQueue<false>();
			break;
		case THREAD_SERIALIZED:
			the_queue = new ThreadQueue<true>();
			break;
#endif
		default:
			throw std::runtime_error("Queue type not enabled");
	}
	*queue = (MPIX_Queue) the_queue;
    
	return MPIX_SUCCESS;
}
}