#include "helpers.hpp"

MPIS_Queue ACTIVE_QUEUE = MPIS_QUEUE_NULL;

std::map<void*, std::function<void()>> deletors;