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
}

CudaQueueEntry::~CudaQueueEntry()
{
    check_gpu(cudaFreeHost(start_location));
    check_gpu(cudaFreeHost(wait_location));
}

void CudaQueueEntry::start_host()
{
    Print::out("Starting host for an entry!");
    while ((*start_location) != threshold)
    {
        std::this_thread::yield();
    }
    Print::out("CUDA Host done!");
    // Call parent method to launch MPI stuff
    QueueEntry::start_host();
}

bool CudaQueueEntry::done()
{
    // Call parent method to figure out if MPI Request is done
    bool value = QueueEntry::done();
    if (value)
    {
        Print::out("Done waiting on MPI! Signaling GPU:", threshold);
        (*wait_location) = threshold;
    }
    return value;
}

void CudaQueueEntry::start_gpu(void* the_stream)
{
    threshold++;
    cudaStream_t* cuda_stream = (cudaStream_t*)the_stream;
    CUstream      real_stream = *cuda_stream;
    Print::out("Starting asking GPU to write:", threshold);
    force_gpu(cuStreamWriteValue64(real_stream, start_dev, threshold, 0));
}

void CudaQueueEntry::wait_gpu(void* the_stream)
{
    Print::out("Asked GPU to wait for: ", threshold);
    cudaStream_t* cuda_stream = (cudaStream_t*)the_stream;
    CUstream      real_stream = *cuda_stream;
    force_gpu(cuStreamWaitValue64(real_stream, wait_dev, threshold, 0));
}

CudaQueue::CudaQueue(cudaStream_t* stream)
    : thr(&CudaQueue::progress, this), my_stream(stream), wait_cntr(0)
{
    // force_gpu(cuInit(0));
    // force_gpu(cudaSetDevice(0));
}

CudaQueue::~CudaQueue()
{
    shutdown = true;
    thr.join();
}

void CudaQueue::progress()
{
    check_gpu(cudaSetDevice(0));
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