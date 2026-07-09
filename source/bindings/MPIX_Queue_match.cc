#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Queue_match(MPIS_Queue queue, MPIS_Request* request, MPI_Status* status)
{
    using namespace Communication;
    std::shared_ptr<Request>* internal_request =
        convert_request_ptr(request, RequestState::UNMATCHED);
    Queue* the_queue = (Queue*)(queue);

    the_queue->initiate_match({*internal_request});
    (*request)->state = RequestState::MATCHED;
    the_queue->finalize_match({*internal_request});

    return MPIS_SUCCESS;
}
}