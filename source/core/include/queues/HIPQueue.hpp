#ifndef ST_HIP_QUEUE
#define ST_HIP_QUEUE

#include <hip/hip_runtime.h>

#include <memory>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"
#include "misc/mem_pool.hpp"

/**
 * @brief
 * @details
 */
class HIPQueueEntry : public QueueEntry
{
public:
    /**
     * @brief
     * @param [in] qe
     * @param [in] start
     * @param [in] wait
     */
    HIPQueueEntry(std::shared_ptr<Request> qe, Progress::CounterType* start,
                  Progress::CounterType* wait);
    ~HIPQueueEntry() = default;

    /**
     * @brief
     * @param [in] stream
     */
    void start_gpu(void* stream) override;
    /**
     * @brief
     * @param [in] stream
     */
    void wait_gpu(void* stream) override;

protected:
    void* start_dev;  //!<
    void* wait_dev;   //!<

private:
    /** @brief */
    void initialize_device_ptrs();
};

/**
 * @brief
 * @details
 */
class HIPQueue : public Queue
{
public:
    /**
     * @brief
     * @param [in] stream
     */
    HIPQueue(hipStream_t* stream);
    ~HIPQueue() = default;

    /**
     * @brief
     * @param [in, out] req
     */
    void enqueue_operation(std::shared_ptr<Request> req) override;
    /** @brief */
    void enqueue_waitall() override;

protected:
    hipStream_t*    my_stream;        //!<
    Memory::GPUPool completion_pool;  //!<

private:
    /** @brief
     * @return
     */
    static Memory::gpu_host_memory_resource* get_raw_allocator()
    {
        static Memory::gpu_host_memory_resource the_resource;
        return &the_resource;
    }

    /** @brief
     * @return
     */
    Progress::CounterType* alloc_counter_buffer();
};

#endif