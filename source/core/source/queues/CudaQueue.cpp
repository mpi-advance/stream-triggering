#include "queues/CudaQueue.hpp"

void CudaQueue::enqueue_operation(std::shared_ptr<Request> request)
{
    size_t request_id = request->getID();
    /* .contains is a C++ 23 feature not supported by Cuda 12.8 */
    if (!request_cache.contains(request_id))
    {
        request_cache.emplace(request_id, std::make_unique<CudaQueueEntry>(request));
    }

    QueueEntry& cqe = *request_cache.at(request_id);
    progress_engine.enqueued_start(cqe, cqe.increment());
    cqe.start_gpu(my_stream);
    entries.push_back(cqe);
}