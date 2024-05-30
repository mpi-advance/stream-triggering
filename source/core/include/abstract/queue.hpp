#ifndef ST_ABSTRACT_QUEUE
#define ST_ABSTRACT_QUEUE

#include "mpi.h"

#include <stdint.h>
#include <vector>

class QueueEntry
{
public:
	QueueEntry(MPI_Request req) : my_request(req) {}
	virtual ~QueueEntry() = default;

	operator uintptr_t() const
	{
		return (uintptr_t) (*this);
	}

	virtual void prepare() = 0;

	virtual void start() = 0;

	virtual bool done() = 0;

	virtual void progress() = 0;

protected:
	MPI_Request my_request;
};

class Queue
{
public:
	virtual ~Queue() = default;

	virtual QueueEntry *create_entry(MPI_Request) = 0;

	virtual void enqueue_operation(QueueEntry *qe)
	{
		entries.push_back(qe);
	}

	virtual void enqueue_waitall() = 0;

	virtual void host_wait() = 0;

	operator uintptr_t() const
	{
		return (uintptr_t) (*this);
	}

protected:
	std::vector<QueueEntry *> entries;
};

#endif