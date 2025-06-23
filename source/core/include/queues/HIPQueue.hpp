#ifndef ST_HIP_QUEUE
#define ST_HIP_QUEUE

#include <hip/hip_runtime.h>

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

class HIPQueueEntry : public QueueEntry
{
public:
    HIPQueueEntry(std::shared_ptr<Request> qe);
    ~HIPQueueEntry();

    void start_host() override;
    void start_gpu(void *) override;
    void wait_gpu(void *) override;
    bool done() override;

protected:
    int64_t* start_location;
    int64_t* wait_location;

    void* start_dev;
    void* wait_dev;
};

class HIPQueue : public Queue
{
public:
    HIPQueue(hipStream_t*);
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
    std::atomic<int> wait_cntr;

    std::vector<std::reference_wrapper<QueueEntry>> entries;
    std::vector<std::reference_wrapper<QueueEntry>> s_ongoing;
    std::queue<std::reference_wrapper<QueueEntry>>  w_ongoing;

    std::map<size_t, HIPQueueEntry> request_cache;

    void progress();
};

#endif