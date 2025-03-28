#ifndef ST_HIP_QUEUE
#define ST_HIP_QUEUE

#include <hip/hip_runtime.h>

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>

#include "abstract/queue.hpp"
#include "abstract/entry.hpp"

class HIPQueueEntry : public QueueEntry
{
public:
    HIPQueueEntry(std::shared_ptr<Request> qe);
    ~HIPQueueEntry();

    void start() override;
    bool done() override;

    void launch_wait_kernel(hipStream_t);
    void launch_start_kernel(hipStream_t);

protected:
    std::shared_ptr<Request> my_request;
    MPI_Request              mpi_request;
    int64_t*                 start_location;
    int64_t*                 wait_location;

    void* start_dev;
    void* wait_dev;
};

class HIPQueue : public Queue
{
public:
    HIPQueue(hipStream_t *);
    ~HIPQueue();

    void enqueue_operation(std::shared_ptr<Request> qe) override;
    void enqueue_waitall() override;
    void host_wait() override;
    void match(std::shared_ptr<Request> qe) override;

protected:
    hipStream_t* my_stream;

    std::thread thr;
    bool        shutdown = false;

    std::mutex       queue_guard;
    std::atomic<int> start_cntr;
    std::atomic<int> wait_cntr;

    std::vector<HIPQueueEntry*> entries;
    std::vector<HIPQueueEntry*> s_ongoing;
    std::queue<HIPQueueEntry*>  w_ongoing;

    void progress();
};

#endif