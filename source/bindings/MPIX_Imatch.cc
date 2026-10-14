#include "helpers.hpp"

extern "C" {

int MPIS_Imatch(MPIS_Request tomatch, MPIS_Request* newrequest, MPI_Info info,
                MPIS_Request* request)
{
    MPIS_BINDING_ENTER

    int err_code = MPIS_Imatchall(1, &tomatch, newrequest, info, request);

    MPIS_BINDING_EXIT
    return err_code;
}
}
