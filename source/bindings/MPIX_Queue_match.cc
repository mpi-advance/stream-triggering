#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Queue_match(MPIS_Queue queue, MPIS_Request* request, MPI_Status* status)
{
    using namespace Communication;
    std::shared_ptr<Request>* internal_request =
        convert_request_ptr(request, RequestState::UNMATCHED);
    Queue* the_queue = (Queue*)(queue);

    the_queue->match(*internal_request);
    (*request)->state = RequestState::MATCHED;
    (*internal_request)->wait_on_match();

    return MPIS_SUCCESS;
}
}