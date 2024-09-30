#include "stream-triggering.h"

extern "C" {

int MPIS_Prepare_all(int len, MPIS_Request requests[])
{
    for (int index = 0; index < len; index++)
    {
        int err_code = MPIS_Prepare(requests[index]);
        if (MPIS_SUCCESS != err_code)
            return err_code;
    }
    return MPIS_SUCCESS;
}
}