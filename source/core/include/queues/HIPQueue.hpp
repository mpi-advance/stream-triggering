#ifndef ST_HIP_QUEUE
#define ST_HIP_QUEUE

#include <hip/hip_runtime.h>

#include <memory>

#include "abstract/queue.hpp"
#include "misc/mem_pool.hpp"

/**
 * @brief [TODO]
 * @details
 */
class HIPQueue : public Queue
{
public:
    /**
     * @brief [TODO]
     * @param [in] stream
     */
    HIPQueue(hipStream_t* stream);
    /** @brief [TODO] */
    ~HIPQueue();

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
    hipStream_t* my_stream;  //!<
    void*        start_dev;  //!< [TODO]
    void*        wait_dev;   //!< [TODO]
};

#endif
