#ifndef ST_ABSTRACT_QUEUE
#define ST_ABSTRACT_QUEUE

#include "mpi.h"
#include "request.hpp"

#include <stdint.h>
#include <vector>

class Queue
{

public:
	virtual ~Queue() = default;

	virtual void enqueue_operation(Communication::Request *qe) = 0;

	virtual void enqueue_waitall() = 0;

	virtual void host_wait() = 0;

	operator uintptr_t() const
	{
		return (uintptr_t) (*this);
	}
};

#endif