#include "queues/HIPQueue.hpp"

#include "misc/print.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

HIPQueue::HIPQueue(hipStream_t* stream)
    : Queue(Memory::gpu_host_resource()), my_stream(stream)
{
    // force_gpu(hipSetDevice(0));
    force_gpu(hipHostGetDevicePointer(&start_dev, start_signal, 0));
    force_gpu(hipHostGetDevicePointer(&wait_dev, wait_signal, 0));
}

HIPQueue::~HIPQueue()
{
    /* The stream must stop touching the counters before ~Queue frees them */
    check_gpu(hipStreamSynchronize(*my_stream));
}

void HIPQueue::enqueue_startall(std::vector<std::shared_ptr<Request>> requests)
{
    Progress::CounterType target = build_start(requests);
    force_gpu(hipStreamWriteValue64(*my_stream, start_dev, target, 0));
}

void HIPQueue::enqueue_waitall(std::vector<std::shared_ptr<Request>> requests,
                               MPI_Status*                           statuses)
{
    Progress::CounterType target = build_wait(requests, statuses);
    force_gpu(hipStreamWaitValue64(*my_stream, wait_dev, target, 0));
}

void HIPQueue::host_wait()
{
    Queue::host_wait();
    Print::out("Waiting on device!");
    force_gpu(hipStreamSynchronize(*my_stream));
}

bool HIPQueue::queue_test()
{
    if (!Queue::queue_test())
    {
        return false;
    }
    hipError_t status = hipStreamQuery(*my_stream);
    if (hipErrorNotReady == status)
    {
        return false;
    }
    force_gpu(status);
    return true;
}
