#ifndef ST_ABSTRACT_QUEUE
#define ST_ABSTRACT_QUEUE

#include <stdint.h>

#include <memory>
#include <vector>

#include "request.hpp"

using namespace Communication;

class Queue
{
public:
    virtual ~Queue() = default;

    virtual void enqueue_operation(std::shared_ptr<Request> qe) = 0;

    virtual void enqueue_waitall() = 0;

    virtual void host_wait() = 0;

    virtual void match(std::shared_ptr<Request> qe) = 0;

    operator uintptr_t() const
    {
        return (uintptr_t)(*this);
    }
};

#endif