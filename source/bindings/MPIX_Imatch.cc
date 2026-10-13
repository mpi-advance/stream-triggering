#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Imatch(MPIS_Request* request, MPIS_Request* match_request)
{
    MPIS_BINDING_ENTER
    using namespace Communication;
    std::shared_ptr<Request>* internal_request =
        convert_request_ptr(request, {Request::State::UNMATCHED});
    Queue* the_queue = (Queue*)(ACTIVE_QUEUE);

    the_queue->initiate_match({*internal_request});

    (*match_request) = new MPIS_Request_struct{true, (uintptr_t)(internal_request)};

    (*internal_request)->state = Request::State::MATCHING;

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}