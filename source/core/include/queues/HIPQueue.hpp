#ifndef ST_HIP_QUEUE
#define ST_HIP_QUEUE

#include <hip/hip_runtime.h>

#include <memory>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

class HIPQueueEntry : public QueueEntry
{
public:
    HIPQueueEntry(std::shared_ptr<Request> qe);
    ~HIPQueueEntry();

    void start_gpu(void*) override;
    void wait_gpu(void*) override;

protected:
    void* start_dev;
    void* wait_dev;
};

class HIPQueue : public Queue
{
public:
    HIPQueue(hipStream_t*);
    ~HIPQueue() = default;

    void enqueue_operation(std::shared_ptr<Request> req) override;
    void enqueue_waitall() override;

protected:
    hipStream_t* my_stream;
};

#endif