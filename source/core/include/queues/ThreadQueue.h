#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include "abstract/queue.hpp"

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>

class ThreadQueueEntry : public QueueEntry
{
public:
	ThreadQueueEntry(MPI_Request req) : QueueEntry(req) {}
	~ThreadQueueEntry() = default;

	void prepare() override;

	void start() override;

	bool done() override;

	void progress() override;
};

class ThreadQueue : public Queue
{
public:
	ThreadQueue();
	~ThreadQueue();

	QueueEntry* create_entry(MPI_Request) override;
	void enqueue_operation(QueueEntry *) override;
	void enqueue_waitall() override;
	void host_wait() override;

protected:
	std::atomic<int> busy;
	std::thread thr;
	bool        shutdown = false;
	std::mutex  queue_guard;

	std::queue<size_t>      stop_counts;
	std::vector<QueueEntry *> ongoing;

	void progress();
};

#endif