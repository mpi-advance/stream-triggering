#ifndef ST_CUDA_QUEUE
#define ST_CUDA_QUEUE

#include "abstract/queue.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>

class CudaQueueEntry : public QueueEntry
{
public:
	CudaQueueEntry(MPI_Request req);
	~CudaQueueEntry();

	void prepare() override;

	void start() override;

	bool done() override;

	void progress() override;

	void launch_wait_kernel(CUstream);
	void launch_start_kernel(CUstream);

protected:
	int *start_location;
	int *wait_location;

	CUdeviceptr start_dev;
	CUdeviceptr wait_dev;
};

class CudaQueue : public Queue
{
public:
	CudaQueue(); 
	~CudaQueue();

	QueueEntry *create_entry(MPI_Request) override;
	void        enqueue_operation(QueueEntry *qe) override;
	void        enqueue_waitall() override;
	void        host_wait() override;

protected:
    cudaStream_t my_stream;

	std::atomic<int> busy;
	std::thread      thr;
	bool             shutdown = false;

	std::mutex                queue_guard;

	std::vector<CudaQueueEntry *> s_ongoing;
	std::queue<CudaQueueEntry *> w_ongoing;

	void progress();
};

#endif