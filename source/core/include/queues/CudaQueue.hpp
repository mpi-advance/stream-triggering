#ifndef ST_CUDA_QUEUE
#define ST_CUDA_QUEUE

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>

#include "abstract/queue.hpp"
#include "misc/mem_pool.hpp"

/**
 * @brief [TODO]
 * @details
 */
class CudaQueue : public Queue
{
public:
    /**
     * @brief [TODO]
     * @param [in] stream
     */
    CudaQueue(cudaStream_t* stream);
    /** @brief [TODO] */
    ~CudaQueue();

    /**
     * @brief [TODO]
     * @param [in, out] requests
     */
    void enqueue_startall(std::vector<std::shared_ptr<Request>> requests) override;
    /**
     * @brief [TODO]
     * @param [in, out] requests
     * @param [out] statuses
     */
    void enqueue_waitall(std::vector<std::shared_ptr<Request>> requests,
                         MPI_Status*                           statuses) override;
    /**
     * @brief [TODO]
     * @return
     */
    bool queue_test() override;
    /** @brief [TODO] */
    void host_wait() override;

protected:
    cudaStream_t* my_stream;  //!<
    CUdeviceptr   start_dev;  //!< [TODO]
    CUdeviceptr   wait_dev;   //!< [TODO]
};

#endif
