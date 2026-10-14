#include "helpers.hpp"

extern "C" {

int MPIS_Matchall(int len, MPIS_Request tomatch[], MPIS_Request newrequests[],
                  MPI_Info info)
{
    MPIS_BINDING_ENTER

    MPIS_Request match_request;
    int err_code = MPIS_Imatchall(len, tomatch, newrequests, info, &match_request);
    if (MPIS_SUCCESS != err_code)
        return err_code;

    err_code = MPIS_Wait(&match_request, MPI_STATUS_IGNORE);

    MPIS_BINDING_EXIT
    return err_code;
}
}
