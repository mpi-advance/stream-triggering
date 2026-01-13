#ifndef ST_CUDA_QUEUE
#define ST_CUDA_QUEUE

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

class CudaQueueEntry : public QueueEntry
{
public:
    CudaQueueEntry(std::shared_ptr<Request> qe);
    ~CudaQueueEntry();

    void start_gpu(void*) override;
    void wait_gpu(void*) override;

protected:
    CUdeviceptr start_dev;
    CUdeviceptr wait_dev;
};

class CudaQueue : public Queue
{
public:
    CudaQueue(cudaStream_t*);
    ~CudaQueue() = default;

    void enqueue_operation(std::shared_ptr<Request> req) override;
    void enqueue_waitall() override;

protected:
    cudaStream_t* my_stream;
};

#endif