#include "queues/HIPQueue.hpp"

#include "misc/print.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

HIPQueueEntry::HIPQueueEntry(std::shared_ptr<Request> req, Progress::CounterType* start,
                             Progress::CounterType* wait)
    : QueueEntry(req, start, wait)
{
    initialize_device_ptrs();
}

void HIPQueueEntry::initialize_device_ptrs()
{
    force_gpu(hipHostGetDevicePointer(&start_dev, start_location, 0));
    force_gpu(hipHostGetDevicePointer(&wait_dev, wait_location, 0));
}

void HIPQueueEntry::start_gpu(void* the_stream)
{
    // Print::out("Starting asking GPU to write:", threshold);
    force_gpu(
        hipStreamWriteValue64(*((hipStream_t*)the_stream), start_dev, threshold, 0));
}

void HIPQueueEntry::wait_gpu(void* the_stream)
{
    // Print::out("Asked GPU to wait for: ", threshold);
    force_gpu(hipStreamWaitValue64(*((hipStream_t*)the_stream), wait_dev, threshold, 0));
}

HIPQueue::HIPQueue(hipStream_t* stream)
    : my_stream(stream), completion_pool(get_raw_allocator())
{
    // force_gpu(hipSetDevice(0));
}

void HIPQueue::enqueue_operation(std::shared_ptr<Request> request)
{
    size_t request_id = request->getID();
    if (!request_cache.contains(request_id))
    {
        request_cache.emplace(
            request_id, std::make_unique<HIPQueueEntry>(request, alloc_counter_buffer(),
                                                        alloc_counter_buffer()));
    }

    QueueEntry& cqe = *request_cache.at(request_id);
    cqe.increment();
    progress_engine.enqueued_start(cqe);
    cqe.start_gpu(my_stream);
    entries.push_back(cqe);
}

void HIPQueue::enqueue_waitall()
{
    // Make sure there's at least one thing to wait on
    if(0 == entries.size())
    {
        Print::out("No Requests have been started!");
        return;
    }

    // Use the first queue entry to try to re-use completion buffers
    QueueEntry& first = entries.at(0);
    // If there's only one entry, we can just use a "WaitEntry"
    if(1 == entries.size())
    {
        progress_engine.enqueued_wait(first);
        first.wait_gpu(my_stream);
        entries.clear();
        return;
    }

    // Otherwise let us collect all the MPI_Requests for a WaitallEntry
    std::vector<Progress::RequestType> reqs;
    for (QueueEntry& entry : entries)
    {
        reqs.push_back(entry.get_mpi_request());
    }

    progress_engine.enqueued_wait(first.convert_to_waitall(reqs));
    first.wait_gpu(my_stream);
    entries.clear();
}

Progress::CounterType* HIPQueue::alloc_counter_buffer()
{
    return (Progress::CounterType*)completion_pool.allocate(
        sizeof(Progress::CounterType));
}