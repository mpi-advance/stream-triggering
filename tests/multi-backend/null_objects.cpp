#include "../common/checks.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    Checks::init(&argc, &argv, 1);

    // Test queue
    MPIS_Queue my_queue = MPIS_QUEUE_NULL;
    EXPECT(MPI_SUCCESS == MPIS_Queue_free(&my_queue), "freeing MPIS_QUEUE_NULL failed");
    EXPECT(MPI_SUCCESS == MPIS_Queue_free(nullptr),
           "freeing a null queue pointer failed");

    // Test request
    MPIS_Request my_req = MPIS_REQUEST_NULL;
    EXPECT(MPI_SUCCESS == MPIS_Request_free(&my_req), "freeing MPIS_REQUEST_NULL failed");
    EXPECT(MPI_SUCCESS == MPIS_Request_free(nullptr),
           "freeing a null request pointer failed");

    return Checks::finish("null_objects");
}