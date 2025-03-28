#include "queues/HIPQueue.hpp"

#include "abstract/match.hpp"
#include "misc/print.hpp"
#include "safety/hip.hpp"
#include "safety/mpi.hpp"

HIPQueueEntry::HIPQueueEntry(std::shared_ptr<Request> req) : QueueEntry(req)
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

void HIPQueueEntry::start()
{
    while ((*start_location) != 1)
    {
        std::this_thread::yield();
    }
    // Call parent method to launch MPI stuff
    QueueEntry::start();
}

bool HIPQueueEntry::done()
{
    // Call parent method to figure out if MPI Request is done
    bool value = QueueEntry::done();
    if (value)
    {
        (*wait_location) = 1;
    }
    return value;
}

void HIPQueueEntry::launch_start_kernel(hipStream_t the_stream)
{
    force_hip(hipStreamWriteValue64(the_stream, start_dev, 1, 0));
}

void HIPQueueEntry::launch_wait_kernel(hipStream_t the_stream)
{
    force_hip(hipStreamWaitValue64(the_stream, wait_dev, 1, 0));
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
        while (start_cntr.load() > 0 || wait_cntr.load() > 0)
        {
            {
                std::scoped_lock<std::mutex> incoming_lock(queue_guard);
                for (HIPQueueEntry* entry : s_ongoing)
                {
                    entry->start();
                    start_cntr--;
                    w_ongoing.push(entry);
                }
                s_ongoing.clear();
            }

            for (size_t i = 0; i < w_ongoing.size(); ++i)
            {
                HIPQueueEntry* entry = w_ongoing.front();
                if (entry->done())
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

void HIPQueue::enqueue_operation(std::shared_ptr<Request> qe)
{
    if (wait_cntr.load() > 0)
        Print::out("WARNING!");

    HIPQueueEntry* cqe = new HIPQueueEntry(qe);
    cqe->launch_start_kernel(*my_stream);
    start_cntr++;
    entries.push_back(cqe);

    std::scoped_lock<std::mutex> incoming_lock(queue_guard);
    s_ongoing.push_back(cqe);
}

void HIPQueue::enqueue_waitall()
{
    while (start_cntr.load())
    {
        // Do nothing
    }

    for (HIPQueueEntry* entry : entries)
    {
        entry->launch_wait_kernel(*my_stream);
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
    while (start_cntr.load() || wait_cntr.load())
    {
        // Do nothing.
    }
}

void HIPQueue::match(std::shared_ptr<Request> request)
{
    // Normal matching
    Communication::BlankMatch();
    request->toggle_match();
}