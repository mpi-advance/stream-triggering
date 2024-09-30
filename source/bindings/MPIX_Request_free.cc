#include "helpers.hpp"

extern "C" {

int MPIS_Request_free(MPIS_Request *request)
{
    using namespace Communication;
	std::shared_ptr<Request>* the_request = convert_request(request);
	delete the_request;

	*request = MPIS_REQUEST_NULL;
	return MPIS_SUCCESS;
}
}