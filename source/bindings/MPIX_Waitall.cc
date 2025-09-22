#include "stream-triggering.h"

extern "C" {

int MPIS_Waitall(int len, MPIS_Request requests[], MPI_Status statuses[])
{
    for (int i = 0; i < len; ++i)
    {
        // Status is unsupported
        int err_code = MPIS_Wait(&requests[i], MPI_STATUS_IGNORE);
        if (MPIS_SUCCESS != err_code)
            return err_code;
    }
    return MPIS_SUCCESS;
}
}