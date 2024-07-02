#ifndef ST_HPE_QUEUE
#define ST_HPE_QUEUE

#include "abstract/queue.hpp"

#include <hip/hip_runtime.h>

class HPEQueueEntry : public QueueEntry
{
public:
	HPEQueueEntry(MPI_Request req);
	~HPEQueueEntry();

	void prepare() override;

	void start() override;

	bool done() override;

	void progress() override;
};

class HPEQueue : public Queue
{
public:
	HPEQueue(hipStream_t *stream_addr);
    ~HPEQueue();

	QueueEntry *create_entry(MPI_Request) override;
	void        enqueue_waitall() override;
	void        host_wait() override;

protected:
	hipStream_t *my_stream;

    MPI_Comm dup_comm;
    MPIX_Queue my_queue;
};

#endif