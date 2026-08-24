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

    virtual void initiate_match(std::vector<std::shared_ptr<Request>> requests)
    {
        for (auto& req : requests)
        {
            if (Operation::BARRIER > req->operation)
            {
                // Normal matching
                Match::Blank::match(*req);
            }
        }
    }

    virtual void finalize_match(std::vector<std::shared_ptr<Request>> requests)
    {
        std::vector<MPI_Request> request_train;
        std::vector<MPI_Status>  status_train;
        for (auto& req : requests)
        {
            if (Operation::BARRIER > req->operation)
            {
                req->join_waitall_match(request_train, status_train);
                Print::out("Train size now:", request_train.size());
            }
        }

        force_mpi(
            MPI_Waitall(request_train.size(), request_train.data(), status_train.data()));

        Print::out("Done with all matching.");
        for (auto& req : requests)
        {
            req->set_match();
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