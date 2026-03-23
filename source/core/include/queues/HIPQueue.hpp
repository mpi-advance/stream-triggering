#ifndef ST_HIP_QUEUE
#define ST_HIP_QUEUE

#include <hip/hip_runtime.h>

#include <memory>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"
#include "misc/mem_pool.hpp"

class HIPQueueEntry : public QueueEntry
{
public:
    HIPQueueEntry(std::shared_ptr<Request> qe, Progress::CounterType* start,
                  Progress::CounterType* wait);
    ~HIPQueueEntry() = default;

    void start_gpu(void*) override;
    void wait_gpu(void*) override;

protected:
    void* start_dev;
    void* wait_dev;

private:
    void initialize_device_ptrs();
};

class HIPQueue : public Queue
{
public:
    HIPQueue(hipStream_t*);
    ~HIPQueue() = default;

    void enqueue_operation(std::shared_ptr<Request> req) override;
    void enqueue_waitall() override;

protected:
    hipStream_t*    my_stream;
    Memory::GPUPool completion_pool;

private:
    static Memory::gpu_host_memory_resource* get_raw_allocator()
    {
        static Memory::gpu_host_memory_resource the_resource;
        return &the_resource;
    }

    Progress::CounterType* alloc_counter_buffer();
};

#endif