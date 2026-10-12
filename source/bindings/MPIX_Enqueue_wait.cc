#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Enqueue_wait(MPIS_Queue queue, MPIS_Request* request, MPI_Status* status)
{
    MPIS_BINDING_ENTER
    using namespace Communication;
    Queue* the_queue = (Queue*)(queue);

    std::shared_ptr<Request>* internal_request =
        convert_request_ptr(request, RequestState::MATCHED);

    MPI_Status* out_status =
        (MPI_STATUS_IGNORE == status || nullptr == status) ? nullptr : status;
    the_queue->enqueue_wait(*internal_request, out_status);

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}
