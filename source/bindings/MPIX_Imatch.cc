#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Imatch(MPIS_Request* request, MPIS_Request* match_request)
{
    using namespace Communication;
    std::shared_ptr<Request>* internal_request =
        convert_request_ptr(request, RequestState::UNMATCHED);
    Queue* the_queue = (Queue*)(ACTIVE_QUEUE);

    the_queue->match(*internal_request);

    (*match_request) =
        new MPIS_Request_struct{RequestState::ONGOING, (uintptr_t)(internal_request)};

    (*request)->state = RequestState::MATCHED;

    return MPIS_SUCCESS;
}
}