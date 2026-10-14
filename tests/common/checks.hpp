#ifndef ST_TEST_CHECKS
#define ST_TEST_CHECKS

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>

#include "queue_setup.hpp"

/* Helpers for tests that check API behavior and buffer contents, and report pass/fail
 * through their exit code. By default a test needs an even number of ranks: rank r talks
 * to rank (r ^ 1). Environment settings:
 *   ST_TEST_SIZE     Elements (ints) per buffer, for tests that move data
 *   ST_TEST_ITERS    Iterations, for tests that loop
 *   ST_TEST_VERBOSE  1 for the tests' informational output
 * Tests fall back to their own defaults, and finish() prints the values used. */

/* CXI point-to-point operations never report MPI statuses (they come back empty) */
#if defined(CXI_BACKEND)
constexpr bool P2P_REPORTS_STATUS = false;
#else
constexpr bool P2P_REPORTS_STATUS = true;
#endif

namespace Checks
{
inline int  rank     = -1;
inline int  size     = -1;
inline int  failures = 0;
inline bool verbose  = false;

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

inline void init(int* argc, char*** argv, int rank_multiple = 2)
{
    int mode;
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &mode);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (0 != size % rank_multiple)
    {
        if (0 == rank)
            fprintf(stderr, "This test needs a multiple of %d ranks.\n", rank_multiple);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    const char* verbose_env = getenv("ST_TEST_VERBOSE");
    verbose = (nullptr != verbose_env) && (0 != strcmp(verbose_env, "")) &&
              (0 != strcmp(verbose_env, "0"));
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

/* Counts buffer elements that don't match base + stride * index, printing the first few
 */
__global__ void check_linear(volatile int* buffer, int buffer_len, int base, int stride,
                             int* mismatches, int rank, int check_id)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= buffer_len)
        return;

    int expected = base + stride * index;
    if (buffer[index] != expected)
    {
        if (atomicAdd(mismatches, 1) < 4)
        {
            printf("[rank %d] FAIL check %d: [%d] is %d, expected %d\n", rank, check_id,
                   index, buffer[index], expected);
        }
    }
}

inline int* gpu_mismatches = nullptr;
inline int  num_checks     = 0;

/* Checks a GPU buffer in stream order: every element must be base + stride * index.
 * Mismatches are collected by finish(). */
inline void expect_buffer(StreamType stream, void* buffer, int buffer_len, int base,
                          int stride)
{
    if (nullptr == gpu_mismatches)
    {
#if defined(NEED_HIP)
        force_gpu(hipMalloc(&gpu_mismatches, sizeof(int)));
        force_gpu(hipMemset(gpu_mismatches, 0, sizeof(int)));
#elif defined(NEED_CUDA)
        force_gpu(cudaMalloc(&gpu_mismatches, sizeof(int)));
        force_gpu(cudaMemset(gpu_mismatches, 0, sizeof(int)));
#endif
    }

    check_linear<<<Common::num_blocks(buffer_len), Common::BLOCK_SIZE, 0, stream>>>(
        (int*)buffer, buffer_len, base, stride, gpu_mismatches, rank, num_checks++);
}

/* A test parameter read from the environment, with the test's default */
struct Param
{
    const char* name;
    const char* env;
    int         value    = -1;     // -1: this test never asked for it
    bool        from_env = false;  // false: the test's default was used
};
inline Param size_param{"size", "ST_TEST_SIZE"};
inline Param iters_param{"iterations", "ST_TEST_ITERS"};

inline int read_param(Param& param, int default_value)
{
    param.value         = default_value;
    const char* env_val = getenv(param.env);
    if (nullptr != env_val && '\0' != env_val[0])
    {
        char* end = nullptr;
        long  val = strtol(env_val, &end, 10);
        if ('\0' != *end || val <= 0 || val > INT_MAX)
        {
            fprintf(stderr, "%s=\"%s\" is not a positive integer.\n", param.env, env_val);
            exit(1);  // May run before MPI_Init, so no MPI_Abort
        }
        param.value    = (int)val;
        param.from_env = true;
    }
    return param.value;
}

/* Elements (ints) per buffer, and iterations, for tests that loop */
inline int buffer_size(int default_value)
{
    return read_param(size_param, default_value);
}
inline int iterations(int default_value)
{
    return read_param(iters_param, default_value);
}

/* Report the result for this rank and return the process exit code */
inline int finish(const char* test_name)
{
    if (nullptr != gpu_mismatches)
    {
        int mismatches = 0;
        device_sync();
#if defined(NEED_HIP)
        check_gpu(
            hipMemcpy(&mismatches, gpu_mismatches, sizeof(int), hipMemcpyDeviceToHost));
        check_gpu(hipFree(gpu_mismatches));
#elif defined(NEED_CUDA)
        check_gpu(
            cudaMemcpy(&mismatches, gpu_mismatches, sizeof(int), cudaMemcpyDeviceToHost));
        check_gpu(cudaFree(gpu_mismatches));
#endif
        gpu_mismatches = nullptr;
        if (0 != mismatches)
        {
            printf("[rank %d] FAIL %d mismatched buffer element(s)\n", rank, mismatches);
            failures += mismatches;
        }
    }

    /* Result line, then the parameters the test used ('*': from the environment) */
    printf("[rank %d] %s: %s (%d failure(s))", rank, test_name,
           (0 == failures) ? "PASS" : "FAIL", failures);
    bool listed = false;
    for (const Param* param : {&size_param, &iters_param})
    {
        if (-1 == param->value)
            continue;
        printf("%s%s=%d%s", listed ? ", " : " [", param->name, param->value,
               param->from_env ? "*" : "");
        listed = true;
    }
    printf("%s\n", listed ? "]" : "");
    MPI_Finalize();
    return (0 == failures) ? 0 : 1;
}

/* Objects most tests share, made by setup() and freed by teardown(). Tests that use
 * them have `using namespace Checks;` and refer to these, rank and size without a
 * prefix; calls into Checks (and Checks::verbose) keep the Checks:: prefix. */
inline MPI_Info   mem_info = MPI_INFO_NULL;
inline void*      send_buf = nullptr;
inline void*      recv_buf = nullptr;
inline StreamType my_stream;
inline MPIS_Queue my_queue;

/* init(), then GPU send/recv buffers of buffer_len ints (send: 100 * rank + index,
 * recv: -1), a stream, and a queue for the backend under test */
inline void setup(int* argc, char*** argv, int buffer_len, int rank_multiple = 2)
{
    init(argc, argv, rank_multiple);

    mem_info = Common::make_gpu_mem_info();
    MPIS_Alloc_mem(sizeof(int) * buffer_len, mem_info, &send_buf);
    MPIS_Alloc_mem(sizeof(int) * buffer_len, mem_info, &recv_buf);

    init_buffers2<<<Common::num_blocks(buffer_len), Common::BLOCK_SIZE>>>(
        (int*)send_buf, (int*)recv_buf, buffer_len, rank);
    device_sync();

    my_stream = Setup::create_stream();
    Setup::create_queue(&my_queue, &my_stream);
}

/* Waits for the queue and any final checks, frees what setup() made, and returns
 * finish()'s exit code. The test frees its own requests first. */
inline int teardown(const char* test_name)
{
    MPIS_Queue_wait(&my_queue);
    device_sync();
    MPIS_Free_mem(send_buf);
    MPIS_Free_mem(recv_buf);
    MPI_Info_free(&mem_info);
    MPIS_Queue_free(&my_queue);
    return finish(test_name);
}
}  // namespace Checks

#endif
