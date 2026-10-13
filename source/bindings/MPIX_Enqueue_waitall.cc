#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

int MPIS_Enqueue_waitall(MPIS_Queue queue, int len, MPIS_Request requests[],
                         MPI_Status statuses[])
{
    MPIS_BINDING_ENTER
    using namespace Communication;
    Queue*                                the_queue = (Queue*)(queue);
    std::vector<std::shared_ptr<Request>> all_requests(len);

    for (int i = 0; i < len; ++i)
    {
        all_requests[i] = *convert_request_ptr(
            &requests[i], {Request::State::MATCHED, Request::State::ACTIVE});
    }

    check_unique_requests(all_requests);

    MPI_Status* out_statuses =
        (MPI_STATUSES_IGNORE == statuses || nullptr == statuses) ? nullptr : statuses;
    the_queue->enqueue_waitall(all_requests, out_statuses);
    for (auto& req : all_requests)
    {
        req->state = Request::State::MATCHED;
    }

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}