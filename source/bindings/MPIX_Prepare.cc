#include "helpers.hpp"

extern "C" {

int MPIS_Prepare(MPIS_Request request)
{
    using namespace Communication;
	std::shared_ptr<Request> the_request = convert_request(request);
    the_request->prepare();
	return MPIS_SUCCESS;
}
}