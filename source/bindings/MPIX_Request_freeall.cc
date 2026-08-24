#include "helpers.hpp"

extern "C" {

int MPIS_Request_freeall(int len, MPIS_Request requests[])
{
    MPIS_BINDING_ENTER

    for (int i = 0; i < len; ++i)
    {
        int err_code = MPIS_Request_free(&requests[i]);
        if (MPIS_SUCCESS != err_code)
            return err_code;
    }

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}