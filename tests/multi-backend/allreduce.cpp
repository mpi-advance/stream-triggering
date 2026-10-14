#include "../common/checks.hpp"
#include "stream-triggering.h"

int main(int argc, char* argv[])
{
    using namespace Checks;

    // Input parameters
    int BUFFER_SIZE = Checks::buffer_size(64);

    // Buffers (send: 100 * rank + index), stream and queue
    Checks::setup(&argc, &argv, BUFFER_SIZE);

    // Make request
    MPIS_Request my_req;
    MPIS_Allreduce_init(send_buf, recv_buf, BUFFER_SIZE, MPI_INT, MPI_SUM, MPI_COMM_WORLD,
                        MPI_INFO_NULL, &my_req);
    Common::match_in_place(1, &my_req);
    MPI_Barrier(MPI_COMM_WORLD);

    if (Checks::verbose)
        std::cout << rank << " starting test!" << std::endl;

    double start = MPI_Wtime();
    MPIS_Enqueue_start(my_queue, &my_req);
    MPIS_Enqueue_wait(my_queue, &my_req, MPI_STATUS_IGNORE);
    MPIS_Queue_wait(&my_queue);
    double end = MPI_Wtime();

    if (Checks::verbose)
    {
        std::cout << rank << " done with GPU communication" << std::endl;
        std::cout << rank << " time: " << end - start << std::endl;
    }

    // Final check: sum over all ranks of (100 * rank + index)
    Checks::expect_buffer(my_stream, recv_buf, BUFFER_SIZE, 100 * (size * (size - 1) / 2),
                          size);

    // Cleanup
    MPIS_Request_freeall(1, &my_req);
    return Checks::teardown("allreduce");
}