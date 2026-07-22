#include "../common/common.hpp"

#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    int mode;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Test queue
    MPIS_Queue my_queue = MPIS_QUEUE_NULL;
    MPIS_Queue_free(&my_queue);
    MPIS_Queue_free(nullptr);

    // Test request
    MPIS_Request my_req = MPIS_REQUEST_NULL;
    MPIS_Request_free(&my_req);
    MPIS_Request_free(nullptr);

    std::cout << rank << " is done." << std::endl;

    MPI_Finalize();

    return 0;
}