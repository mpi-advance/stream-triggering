#include "helpers.hpp"
#include "stream-triggering.h"

extern "C" {

int MPIS_Free_mem(void* baseptr)
{
    deletors.at(baseptr)();

    return MPIS_SUCCESS;
}
}