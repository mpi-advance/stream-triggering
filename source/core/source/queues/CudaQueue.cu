#include "misc/print.hpp"
#include "queues/CudaQueue.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

CudaQueueEntry::CudaQueueEntry(std::shared_ptr<Request> req) : QueueEntry(req)
{
    force_cuda(cuMemHostAlloc(
        (void**)&start_location, sizeof(int64_t),
        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED));
    *start_location = 0;
    force_cuda(cuMemHostGetDevicePointer(&start_dev, start_location, 0));
    force_cuda(cudaHostAlloc(
        (void**)&wait_location, sizeof(int64_t),
        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED));
    *wait_location = 0;
    force_cuda(cuMemHostGetDevicePointer(&wait_dev, wait_location, 0));
}

CudaQueueEntry::~CudaQueueEntry()
{
    check_cuda(cudaFreeHost(start_location));
    check_cuda(cudaFreeHost(wait_location));
}

void CudaQueueEntry::start_host()
{
    while ((*start_location) != threshold)
    {
        std::this_thread::yield();
    }
    // Call parent method to launch MPI stuff
    QueueEntry::start_host();
}

bool CudaQueueEntry::done()
{
    // Call parent method to figure out if MPI Request is done
    bool value = QueueEntry::done();
    if (value)
    {
        (*wait_location) = threshold;
    }
    return value;
}

void CudaQueueEntry::start_gpu(void* the_stream)
{
    cudaStream_t* cuda_stream = (cudaStream_t*)the_stream;
    CUstream      real_stream = *cuda_stream;
    force_cuda(cuStreamWriteValue64(real_stream, start_dev, threshold, 0));
}

void CudaQueueEntry::wait_gpu(void* the_stream)
{
    cudaStream_t* cuda_stream = (cudaStream_t*)the_stream;
    CUstream      real_stream = *cuda_stream;
    force_cuda(cuStreamWaitValue64(real_stream, wait_dev, threshold, 0));
}

CudaQueue::CudaQueue(cudaStream_t* stream)
    : thr(&CudaQueue::progress, this), my_stream(stream), wait_cntr(0)
{
    // force_cuda(cuInit(0));
    // force_cuda(cudaSetDevice(0));
}

CudaQueue::~CudaQueue()
{
    shutdown = true;
    thr.join();
}

void CudaQueue::progress()
{
    check_cuda(cudaSetDevice(0));
    while (!shutdown)
    {
        while (s_ongoing.size() > 0 || wait_cntr.load() > 0)
        {
            if (s_ongoing.size() > 0)
            {
                std::scoped_lock<std::mutex> incoming_lock(queue_guard);
                for (QueueEntry& entry : s_ongoing)
                {
                    entry.start_host();
                    w_ongoing.push(entry);
                }
                s_ongoing.clear();
            }

            for (size_t i = 0; i < w_ongoing.size(); ++i)
            {
                QueueEntry& entry = w_ongoing.front();
                if (entry.done())
                {
                    wait_cntr--;
                    w_ongoing.pop();
                }
                else
                {
                    break;
                }
            }

            if (shutdown)
                break;
        }

        std::this_thread::yield();
    }
}

void CudaQueue::enqueue_waitall()
{
    while (s_ongoing.size())
    {
        // Do nothing
    }

    for (QueueEntry& entry : entries)
    {
        entry.wait_gpu(my_stream);
        wait_cntr++;
        while (wait_cntr.load())
        {
            // do nothing
        }
    }
    entries.clear();
}

void CudaQueue::host_wait()
{
    while (s_ongoing.size() || wait_cntr.load())
    {
        // Do nothing.
    }
}