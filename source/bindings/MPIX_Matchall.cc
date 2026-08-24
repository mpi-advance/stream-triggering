#include <vector>

#include "helpers.hpp"
extern "C" {

int MPIS_Matchall(int len, MPIS_Request requests[], MPI_Status statuses[])
{
    MPIS_BINDING_ENTER

    std::vector<MPIS_Request> imatch_reqs(len);
    for (int i = 0; i < len; ++i)
    {
        int err_code = MPIS_Imatch(&requests[i], &imatch_reqs[i]);
        if (MPIS_SUCCESS != err_code)
            return err_code;
    }

    int err_code = MPIS_Waitall(len, imatch_reqs.data(), statuses);
    if (MPIS_SUCCESS != err_code)
        return err_code;

    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}