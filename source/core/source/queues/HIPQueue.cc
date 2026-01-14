#include "queues/HIPQueue.hpp"

#include "misc/print.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

HIPQueueEntry::HIPQueueEntry(std::shared_ptr<Request> req) : QueueEntry(req)
{
    force_gpu(hipHostMalloc((void**)&start_location, sizeof(int64_t), 0));
    *start_location = 0;
    force_gpu(hipHostGetDevicePointer(&start_dev, start_location, 0));
    force_gpu(hipHostMalloc((void**)&wait_location, sizeof(int64_t), 0));
    *wait_location = 0;
    force_gpu(hipHostGetDevicePointer(&wait_dev, wait_location, 0));

    initialize_lambdas();
}

HIPQueueEntry::~HIPQueueEntry()
{
    check_gpu(hipHostFree(start_location));
    check_gpu(hipHostFree(wait_location));
}

void HIPQueueEntry::start_gpu(void* the_stream)
{
    Print::out("Starting asking GPU to write:", threshold);
    force_gpu(
        hipStreamWriteValue64(*((hipStream_t*)the_stream), start_dev, threshold, 0));
}

void HIPQueueEntry::wait_gpu(void* the_stream)
{
    Print::out("Asked GPU to wait for: ", threshold);
    force_gpu(hipStreamWaitValue64(*((hipStream_t*)the_stream), wait_dev, threshold, 0));
}

HIPQueue::HIPQueue(hipStream_t* stream) : my_stream(stream)
{
    // force_gpu(hipSetDevice(0));
}

void HIPQueue::enqueue_operation(std::shared_ptr<Request> request)
{
    size_t request_id = request->getID();
    if (!request_cache.contains(request_id))
    {
        request_cache.emplace(request_id, std::make_unique<HipQueueEntry>(request));
    }

    QueueEntry& cqe = *request_cache.at(request_id);
    cqe.increment();
    progress_engine.enqueued_start(cqe);
    cqe.start_gpu(my_stream);
    entries.push_back(cqe);
}

void HIPQueue::enqueue_waitall()
{
    for (QueueEntry& entry : entries)
    {
        progress_engine.enqueued_wait(entry);
        entry.wait_gpu(my_stream);
    }
    entries.clear();
}