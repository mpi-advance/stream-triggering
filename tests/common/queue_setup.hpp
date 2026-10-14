#ifndef ST_TEST_QUEUE_SETUP
#define ST_TEST_QUEUE_SETUP

#define USE_STREAM_TRIGGERING  // For common.hpp to compile with stream triggering stuff
#include "common.hpp"

/* GPU stream and MPIS_Queue setup shared by the multi-backend tests. The queue type
 * follows the <BACKEND>_BACKEND define the test is compiled with. */

#if defined(NEED_HIP)
using StreamType = hipStream_t;
#elif defined(NEED_CUDA)
using StreamType = cudaStream_t;
#endif

namespace Setup
{
inline StreamType create_stream()
{
    StreamType stream;
#if defined(NEED_HIP)
    check_gpu(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
#elif defined(NEED_CUDA)
    check_gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#endif
    return stream;
}

inline void create_queue(MPIS_Queue* queue, StreamType* stream)
{
#if defined(HIP_BACKEND) || defined(CUDA_BACKEND)
    MPIS_Queue_create(queue, GPU_MEM_OPS, MPI_INFO_NULL, stream);
#elif defined(CXI_BACKEND)
    MPIS_Queue_create(queue, CXI, MPI_INFO_NULL, stream);
#elif defined(THREAD_BACKEND)
    MPIS_Queue_create(queue, THREAD, MPI_INFO_NULL, stream);
#endif
}
}  // namespace Setup

#endif
