#include <vector>

#include "abstract/queue.hpp"
#include "helpers.hpp"

extern "C" {

/* Hints for the matched requests still come from the *_init calls */
int MPIS_Imatchall(int len, MPIS_Request tomatch[], MPIS_Request newrequests[], MPI_Info,
                   MPIS_Request* request)
{
    MPIS_BINDING_ENTER

    using namespace Communication;
    Queue* the_queue = (Queue*)(ACTIVE_QUEUE);

    /* Check every original before creating anything */
    std::vector<std::shared_ptr<Request>> originals;
    originals.reserve(len);
    for (int i = 0; i < len; ++i)
    {
        originals.push_back(*convert_request(tomatch[i], {Request::State::UNMATCHED}));
    }

    /* Each new handle gets its own Request; the originals are left untouched */
    auto* matching = new std::vector<std::shared_ptr<Request>>();
    matching->reserve(len);
    for (int i = 0; i < len; ++i)
    {
        std::shared_ptr<Request> new_request = originals[i]->clone();
        matching->push_back(new_request);
        newrequests[i] = new MPIS_Request_struct{
            false, (uintptr_t)(new std::shared_ptr<Request>(new_request))};
    }

    the_queue->initiate_match(*matching);
    for (std::shared_ptr<Request>& new_request : *matching)
    {
        new_request->state = Request::State::MATCHING;
    }

    *request = new MPIS_Request_struct{true, (uintptr_t)(matching)};

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}
