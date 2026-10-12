#ifndef ST_CUDA_QUEUE
#define ST_CUDA_QUEUE

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

/**
 * @brief
 * @details
 */
class CudaQueueEntry : public QueueEntry
{
public:
    /**
     * @brief
     * @param [in] qe
     */
    CudaQueueEntry(std::shared_ptr<Request> qe);
    /** @brief */
    ~CudaQueueEntry();

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
    CUdeviceptr start_dev;  //!<
    CUdeviceptr wait_dev;   //!<
};

/**
 * @brief
 * @details
 */
class CudaQueue : public Queue
{
public:
    /**
     * @brief
     * @param [in] stream
     */
    CudaQueue(cudaStream_t* stream);
    ~CudaQueue() = default;

    /**
     * @brief
     * @param [in, out] req
     */
    void enqueue_operation(std::shared_ptr<Request> req) override;
    /** @brief */
    void enqueue_waitall() override;

protected:
    cudaStream_t* my_stream;  //!<
};

#endif