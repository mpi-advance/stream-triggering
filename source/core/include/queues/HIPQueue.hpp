#ifndef ST_HIP_QUEUE
#define ST_HIP_QUEUE

#include <hip/hip_runtime.h>

#include <memory>

#include "abstract/queue.hpp"
#include "misc/mem_pool.hpp"

/**
 * @file HIPQueue.hpp
 * @brief Code for the implementation of a HIP-memops-based Queue backend.
 */

/** @brief Implementation of a stream-triggered Queue using the HIP mem-ops functions with
 * a progress thread.
 * @details This class gets full stream-triggered semantics at the cost of using a
 * progress thread. This class uses GPU mem-ops to coordinate the GPU stream with the
 * progress thread via two memory (host-pinned) counters.
 */
class HIPQueue : public Queue
{
public:
    /** @brief Constructor for a HIP-based GPU-memops ::Queue implementation.
     * @details Provides Memory::gpu_host_resource() to Queue::Queue(), and also gets the
     * device pointers of Queue::start_signal and Queue::wait_signal.
     * @param [in] stream A pointer to the user-created HIP stream to associate with this
     * ::Queue object. Must remain a valid GPU stream until this object goes away.
     */
    HIPQueue(hipStream_t* stream);

    /** @brief Free the ::HIPQueue object. Synchronizes with stream before returning to
     * avoid freeing resources (e.g. counters) that may be in use by the GPU. */
    ~HIPQueue();

    /** @brief Enqueue a startall operation to the GPU stream.
     * @details This method uses Queue::build_start() to prime the CPU progress thread to
     * be ready to do the corresponding MPI operation(s). This function then takes the
     * returned value to enqueue a GPU memory operation on the user-provided stream that
     * will set Queue::start_signal (via HIPQueue::start_dev) to the return value.
     * @param [in, out] requests The vector of Communication::Request objects to enqueue
     * the start of.
     */
    void enqueue_startall(std::vector<std::shared_ptr<Request>> requests) override;

    /** @brief Enqueue a waitall operation to the GPU stream.
     * @details This method uses Queue::build_wait() to prime the CPU progress thread to
     * be ready to wait on the corresponding MPI_Request(s). The threshold returned from
     * that call is the value used by hipStreamWaitValue64 to block the GPU stream
     * until the value has arrived.
     * @param [in, out] requests The Communication::Request to enqueue the wait for.
     * @param [out] statuses A pointer to the MPI_Status to be filled in upon the
     * completion of the wait. Pointer should be valid until the wait is done. Nullptr is
     * an acceptable value.
     */
    void enqueue_waitall(std::vector<std::shared_ptr<Request>> requests,
                         MPI_Status*                           statuses) override;

    /** @brief Check if the HIPQueue object has an ongoing action on the queue.
     * @details This method first checks Queue::queue_test(). If that returns true, then
     * it checks if the GPU is also inactive before returning.
     * @return True if both the progress thread and the provided GPU stream have no
     * ongoing actions, false otherwise.
     */
    bool queue_test() override;

    /** @brief Wait for the progress thread to be inactive and the GPU stream to be
     * synchronized.
     * @sa Queue::host_wait()
     */
    void host_wait() override;

protected:
    /** @brief The user-provided hipStream_t to use for any GPU actions. */
    hipStream_t* my_stream;
    /** @brief The GPU pointer of Queue::start_signal as it was mapped into GPU space. */
    void* start_dev;
    /** @brief The GPU pointer of Queue::wait_signal as it was mapped into GPU space. */
    void* wait_dev;
};

#endif
