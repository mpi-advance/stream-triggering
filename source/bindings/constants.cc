#include "helpers.hpp"

MPIS_Queue         ACTIVE_QUEUE      = MPIS_QUEUE_NULL;
const uintptr_t    MPIS_QUEUE_NULL   = 0;
const MPIS_Request MPIS_REQUEST_NULL = 0;

std::map<void*, std::function<void()>> deletors;