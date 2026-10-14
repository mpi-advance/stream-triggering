#include "helpers.hpp"

extern "C" {

int MPIS_Match(MPIS_Request tomatch, MPIS_Request* newrequest, MPI_Info info)
{
    MPIS_BINDING_ENTER

    MPIS_Request match_request;
    int          err_code = MPIS_Imatch(tomatch, newrequest, info, &match_request);
    if (MPIS_SUCCESS != err_code)
        return err_code;

    err_code = MPIS_Wait(&match_request, MPI_STATUS_IGNORE);

    MPIS_BINDING_EXIT
    return err_code;
}
}
