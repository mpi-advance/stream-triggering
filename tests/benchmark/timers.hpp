#ifndef BENCHMARK_TIMING
#define BENCHMARK_TIMING

#include "common.hpp"

#include <vector>

namespace Timing
{

#if defined NEED_HIP
using Timings = hipEvent_t;
#else
using Timings = double;
#endif

Timings                        base_timer;
std::vector<Timings>           global_timers;
std::vector<Timings>::iterator curr_iter;

size_t       total_iters;
const size_t num_timings     = 10;
size_t       percent_tracker = 10;

static inline void init_timers(size_t iterations)
{
    total_iters = iterations;
#if defined NEED_HIP
    force_gpu(hipEventCreateWithFlags(&base_timer, hipEventDisableSystemFence));
    global_timers = std::vector<Timings>(num_timings, nullptr);
    for (auto& timer : global_timers)
    {
        force_gpu(hipEventCreateWithFlags(&timer, hipEventDisableSystemFence));
    }
#else
    global_timers = std::vector<Timings>(num_timings, 0.0);
#endif
    curr_iter = global_timers.begin();
}

static inline void set_base_timer()
{
#if defined NEED_HIP
    check_gpu(hipEventRecord(base_timer));
#else
    base_timer = MPI_Wtime();
#endif
}

static inline void print_timers(int rank)
{
    for (size_t index = 0; index < num_timings; ++index)
    {
#if defined NEED_HIP
        float time = 0.0;
        check_gpu(hipEventElapsedTime(&time, base_timer, global_timers[index]));
#else
        double time = global_timers[index] - base_timer;
#endif
        std::cout << "[Rank " << rank << ", Timer "<< index << "]: " << time << std::endl;
    }
}

static inline void free_timers()
{
#if defined NEED_HIP
    check_gpu(hipEventDestroy(base_timer));
    for (auto& timer : global_timers)
    {
        if (timer != nullptr)
        {
            check_gpu(hipEventDestroy(timer));
        }
    }
#endif
}

static inline void add_timer(size_t iter)
{
    iter++;
    if (iter * 100 >= percent_tracker * total_iters)
    {
#if defined NEED_HIP
        check_gpu(hipEventRecord(*curr_iter));
#else
        *curr_iter = MPI_Wtime();
#endif
        percent_tracker += 10;
        curr_iter++;
    }
}

}  // namespace Timing

#endif