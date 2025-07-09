#ifndef ST_CUDA_QUEUE
#define ST_CUDA_QUEUE

#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

class CudaQueueEntry : public QueueEntry
{
public:
    CudaQueueEntry(std::shared_ptr<Request> qe);
    ~CudaQueueEntry();

    void start_host() override;
    void start_gpu(void*) override;
    void wait_gpu(void*) override;
    bool done() override;

protected:
    int64_t* start_location;
    int64_t* wait_location;

    CUdeviceptr start_dev;
    CUdeviceptr wait_dev;
};

class CudaQueue : public Queue
{
public:
    CudaQueue(cudaStream_t*);
    ~CudaQueue();

    void enqueue_operation(std::shared_ptr<Request> req) override;
    void enqueue_startall(std::vector<std::shared_ptr<Request>> reqs) override;
    void enqueue_waitall() override;
    void host_wait() override;

protected:
    cudaStream_t* my_stream;

    std::thread thr;
    bool        shutdown = false;

    std::mutex       queue_guard;
    std::atomic<int> wait_cntr;

    std::vector<std::reference_wrapper<QueueEntry>> entries;
    std::vector<std::reference_wrapper<QueueEntry>> s_ongoing;
    std::queue<std::reference_wrapper<QueueEntry>>  w_ongoing;

    std::map<size_t, CudaQueueEntry> request_cache;

    void progress();
};

#endif