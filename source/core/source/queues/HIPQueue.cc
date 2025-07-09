#include "queues/HIPQueue.hpp"

#include "abstract/match.hpp"
#include "misc/print.hpp"
#include "safety/hip.hpp"
#include "safety/mpi.hpp"

HIPQueueEntry::HIPQueueEntry(std::shared_ptr<Request> req)
    : QueueEntry(req)
{
    force_hip(hipHostMalloc((void**)&start_location, sizeof(int64_t), 0));
    *start_location = 0;
    force_hip(hipHostGetDevicePointer(&start_dev, start_location, 0));
    force_hip(hipHostMalloc((void**)&wait_location, sizeof(int64_t), 0));
    *wait_location = 0;
    force_hip(hipHostGetDevicePointer(&wait_dev, wait_location, 0));
}

HIPQueueEntry::~HIPQueueEntry()
{
    check_hip(hipHostFree(start_location));
    check_hip(hipHostFree(wait_location));
}

void HIPQueueEntry::start_host()
{
    while ((*start_location) != threshold)
    {
        std::this_thread::yield();
    }
    // Call parent method to launch MPI stuff
    QueueEntry::start_host();
}

bool HIPQueueEntry::done()
{
    // Call parent method to figure out if MPI Request is done
    bool value = QueueEntry::done();
    if (value)
    {
        (*wait_location) = threshold;
    }
    return value;
}

void HIPQueueEntry::start_gpu(void* the_stream)
{
    force_hip(hipStreamWriteValue64(*((hipStream_t*)the_stream), start_dev,
                                    threshold, 0));
}

void HIPQueueEntry::wait_gpu(void* the_stream)
{
    force_hip(hipStreamWaitValue64(*((hipStream_t*)the_stream), wait_dev,
                                   threshold, 0));
}

HIPQueue::HIPQueue(hipStream_t* stream)
    : thr(&HIPQueue::progress, this), my_stream(stream)
{
    // force_hip(hipSetDevice(0));
}

HIPQueue::~HIPQueue()
{
    shutdown = true;
    thr.join();
}

void HIPQueue::progress()
{
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

void HIPQueue::enqueue_operation(std::shared_ptr<Request> request)
{
    if (wait_cntr.load() > 0)
        Print::out("WARNING!");

    size_t request_id = request->getID();
    if (!request_cache.contains(request_id))
    {
        request_cache.emplace(request_id, request);
    }

    HIPQueueEntry& cqe = request_cache.at(request_id);
    cqe.start_gpu(my_stream);

    entries.push_back(cqe);
    std::scoped_lock<std::mutex> incoming_lock(queue_guard);
    s_ongoing.push_back(cqe);
}

void HIPQueue::enqueue_startall(std::vector<std::shared_ptr<Request>> reqs)
{
    for(auto& req: reqs)
    {
        enqueue_operation(req);
    }
}

void HIPQueue::enqueue_waitall()
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

void HIPQueue::host_wait()
{
    while (s_ongoing.size() || wait_cntr.load())
    {
        // Do nothing.
    }
}