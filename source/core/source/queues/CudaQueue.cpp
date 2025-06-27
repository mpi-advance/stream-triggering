#include "queues/CudaQueue.hpp"

#include "abstract/match.hpp"

void CudaQueue::enqueue_operation(std::shared_ptr<Request> request)
{
    if (wait_cntr.load() > 0)
        Print::out("WARNING!");

    size_t request_id = request->getID();
    if (!request_cache.contains(request_id))
    {
        request_cache.emplace(request_id, request);
    }

    CudaQueueEntry& cqe = request_cache.at(request_id);
    cqe.start_gpu(my_stream);

    entries.push_back(cqe);
    std::scoped_lock<std::mutex> incoming_lock(queue_guard);
    s_ongoing.push_back(cqe);
}