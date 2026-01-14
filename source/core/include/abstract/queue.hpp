#ifndef ST_ABSTRACT_QUEUE
#define ST_ABSTRACT_QUEUE

#include <stdint.h>

#include <map>
#include <memory>
#include <vector>

#include "entry.hpp"
#include "match.hpp"
#include "progress.hpp"
#include "request.hpp"

using namespace Communication;

class Queue
{
public:
    virtual ~Queue() = default;

    virtual void enqueue_operation(std::shared_ptr<Request> req) = 0;
    virtual void enqueue_startall(std::vector<std::shared_ptr<Request>> reqs)
    {
        for (auto& req : reqs)
        {
            enqueue_operation(req);
        }
    }
    virtual void enqueue_waitall() = 0;

    virtual void host_wait()
    {
        progress_engine.wait_until_empty();
    }

    virtual void match(std::shared_ptr<Request> request)
    {
        if (Operation::BARRIER > request->operation)
        {
            // Normal matching
            MPI_Request* mpi_request = request->get_match_requests(1);
            Communication::BlankMatch::match(request->peer, request->tag, mpi_request);
        }
    }

    operator uintptr_t() const
    {
        return (uintptr_t)(*this);
    }

protected:
    Progress::Engine progress_engine;

    std::vector<std::reference_wrapper<QueueEntry>> entries;
    std::map<size_t, std::unique_ptr<QueueEntry>>   request_cache;
};

#endif