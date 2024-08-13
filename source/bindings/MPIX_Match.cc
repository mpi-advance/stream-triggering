#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Match(MPIS_Request* request)
{
    using namespace Communication;
	Request *the_request = (Request *) (*request);
    MatchPair& the_mp = the_request->match();
    the_mp.wait();

	return MPIS_SUCCESS;
}
}