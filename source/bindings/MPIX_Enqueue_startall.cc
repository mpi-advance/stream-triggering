#include "abstract/request.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Enqueue_startall(MPIS_Queue queue, int len, MPIS_Request requests[])
{
    for (int i = 0; i < len; ++i)
    {
        int err_code = MPIS_Enqueue_start(queue, requests[i]);
        if (MPIS_SUCCESS != err_code)
            return err_code;
    }
    return MPIS_SUCCESS;
}
}