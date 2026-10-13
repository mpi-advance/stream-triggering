#include "misc/print.hpp"
#include "queues/CudaQueue.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

CudaQueue::CudaQueue(cudaStream_t* stream)
    : Queue(Memory::gpu_host_resource()), my_stream(stream)
{
    // force_gpu(cuInit(0));
    // force_gpu(cudaSetDevice(0));
    force_gpu(cuMemHostGetDevicePointer(&start_dev, start_signal, 0));
    force_gpu(cuMemHostGetDevicePointer(&wait_dev, wait_signal, 0));
}

CudaQueue::~CudaQueue()
{
    /* The stream must stop touching the counters before ~Queue frees them */
    check_gpu(cudaStreamSynchronize(*my_stream));
}

void CudaQueue::enqueue_startall(std::vector<std::shared_ptr<Request>> requests)
{
    Progress::CounterType target = build_start(requests);
    force_gpu(cuStreamWriteValue64((CUstream)(*my_stream), start_dev, target, 0));
}

void CudaQueue::enqueue_waitall(std::vector<std::shared_ptr<Request>> requests,
                                MPI_Status*                           statuses)
{
    Progress::CounterType target = build_wait(requests, statuses);
    force_gpu(cuStreamWaitValue64((CUstream)(*my_stream), wait_dev, target, 0));
}

void CudaQueue::host_wait()
{
    Queue::host_wait();
    Print::out("Waiting on device!");
    force_gpu(cudaStreamSynchronize(*my_stream));
}

bool CudaQueue::queue_test()
{
    if (!Queue::queue_test())
    {
        return false;
    }
    cudaError_t status = cudaStreamQuery(*my_stream);
    if (cudaErrorNotReady == status)
    {
        return false;
    }
    force_gpu(status);
    return true;
}
