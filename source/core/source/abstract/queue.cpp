#include "abstract/queue.hpp"

#include "abstract/match.hpp"

/* Done here so that CUDAQueue doesn't see the match header, as that a
 * feature from more recent C++ that CUDA compilers don't seem to handle in
 * cuda file (static_assert(false) is not a compile failure when using
 * templates)
 */
void Queue::match(std::shared_ptr<Request> request)
{
    if (Operation::BARRIER != request->operation)
    {
        // Normal matching
        Communication::BlankMatch::match(request->peer);
    }
    request->toggle_match();
}