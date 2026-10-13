#ifndef ST_TEST_CHECKS
#define ST_TEST_CHECKS

#include <cstdio>
#include <exception>

#include "queue_setup.hpp"

/* Helpers for tests that check API behavior and report pass/fail through their exit
 * code, instead of printing buffer contents. Every test using these needs an even number
 * of ranks: rank r talks to rank (r ^ 1). */

/* CXI point-to-point operations never report MPI statuses (they come back empty) */
#if defined(CXI_BACKEND)
constexpr bool P2P_REPORTS_STATUS = false;
#else
constexpr bool P2P_REPORTS_STATUS = true;
#endif

namespace Checks
{
inline int rank     = -1;
inline int failures = 0;

#define EXPECT(cond, ...)                                                       \
    do                                                                          \
    {                                                                           \
        if (!(cond))                                                            \
        {                                                                       \
            Checks::failures++;                                                 \
            printf("[rank %d] FAIL %s:%d: ", Checks::rank, __FILE__, __LINE__); \
            printf(__VA_ARGS__);                                                \
            printf("\n");                                                       \
        }                                                                       \
    } while (0)

inline void init(int* argc, char*** argv)
{
    int mode;
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &mode);
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (0 != size % 2)
    {
        if (0 == rank)
            fprintf(stderr, "This test needs an even number of ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

/* An "empty" status, as MPI returns for MPI_REQUEST_NULL or inactive requests */
inline bool is_empty_status(MPI_Status& status)
{
    int count = -1;
    MPI_Get_count(&status, MPI_INT, &count);
    return MPI_ANY_SOURCE == status.MPI_SOURCE && MPI_ANY_TAG == status.MPI_TAG &&
           0 == count;
}

/* The bindings currently report misuse by throwing (MPISException) */
template <typename Action>
bool throws(Action&& action)
{
    try
    {
        action();
    }
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}

/* Report the result for this rank and return the process exit code */
inline int finish(const char* test_name)
{
    printf("[rank %d] %s: %s (%d failure(s))\n", rank, test_name,
           (0 == failures) ? "PASS" : "FAIL", failures);
    MPI_Finalize();
    return (0 == failures) ? 0 : 1;
}
}  // namespace Checks

#endif
