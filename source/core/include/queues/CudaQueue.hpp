#ifndef ST_CUDA_QUEUE
#define ST_CUDA_QUEUE

#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>

#include "abstract/queue.hpp"

class CudaQueueEntry
{
public:
    CudaQueueEntry(std::shared_ptr<Request> qe);
    ~CudaQueueEntry();

    void prepare();

    void start();

    bool done();

    void launch_wait_kernel(CUstream);
    void launch_start_kernel(CUstream);

protected:
    std::shared_ptr<Request> my_request;
    MPI_Request              mpi_request;
    int64_t*                 start_location;
    int64_t*                 wait_location;

    CUdeviceptr start_dev;
    CUdeviceptr wait_dev;
};

class CudaQueue : public Queue
{
public:
    CudaQueue(cudaStream_t*);
    ~CudaQueue();

    void enqueue_operation(std::shared_ptr<Request> qe) override;
    void enqueue_waitall() override;
    void host_wait() override;
    void match(std::shared_ptr<Request> qe) override;

protected:
    cudaStream_t* my_stream;

    std::thread thr;
    bool        shutdown = false;

    std::mutex       queue_guard;
    std::atomic<int> start_cntr;
    std::atomic<int> wait_cntr;

    std::vector<CudaQueueEntry*> entries;
    std::vector<CudaQueueEntry*> s_ongoing;
    std::queue<CudaQueueEntry*>  w_ongoing;

    void progress();
};

#endif