#include "queues/CudaQueue.hpp"

#include "abstract/match.hpp"

void CudaQueue::match(std::shared_ptr<Request> request)
{
    // Normal matching
    Communication::BlankMatch();
    request->toggle_match();
}