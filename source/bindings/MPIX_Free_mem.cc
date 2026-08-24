#include "helpers.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Free_mem(void* baseptr)
{
    MPIS_BINDING_ENTER

    deletors.at(baseptr)();
    
    MPIS_BINDING_EXIT
    return MPIS_SUCCESS;
}
}