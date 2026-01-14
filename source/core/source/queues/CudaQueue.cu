#include "misc/print.hpp"
#include "queues/CudaQueue.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

CudaQueueEntry::CudaQueueEntry(std::shared_ptr<Request> req) : QueueEntry(req)
{
    force_gpu(cuMemHostAlloc((void**)&start_location, sizeof(int64_t),
                             CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED));
    *start_location = 0;
    force_gpu(cuMemHostGetDevicePointer(&start_dev, start_location, 0));
    force_gpu(cudaHostAlloc((void**)&wait_location, sizeof(int64_t),
                            CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED));
    *wait_location = 0;
    force_gpu(cuMemHostGetDevicePointer(&wait_dev, wait_location, 0));

    initialize_lambdas();
}

CudaQueueEntry::~CudaQueueEntry()
{
    check_gpu(cudaFreeHost(start_location));
    check_gpu(cudaFreeHost(wait_location));
}

void CudaQueueEntry::start_gpu(void* the_stream)
{
    cudaStream_t* cuda_stream = (cudaStream_t*)the_stream;
    CUstream      real_stream = *cuda_stream;
    Print::out("<E> Starting asking GPU to write:", threshold, start_location);
    force_gpu(cuStreamWriteValue64(real_stream, start_dev, threshold, 0));
}

void CudaQueueEntry::wait_gpu(void* the_stream)
{
    Print::out("<E> Asked GPU to wait for: ", threshold);
    cudaStream_t* cuda_stream = (cudaStream_t*)the_stream;
    CUstream      real_stream = *cuda_stream;
    force_gpu(cuStreamWaitValue64(real_stream, wait_dev, threshold, 0));
}

CudaQueue::CudaQueue(cudaStream_t* stream) : my_stream(stream)
{
    // force_gpu(cuInit(0));
    // force_gpu(cudaSetDevice(0));
}

void CudaQueue::enqueue_waitall()
{
    for (QueueEntry& entry : entries)
    {
        progress_engine.enqueued_wait(entry);
        entry.wait_gpu(my_stream);
    }
    entries.clear();
}