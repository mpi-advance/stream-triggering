
#include "mpi.h"
#include <hip/hip_runtime.h>

#include <iostream>

#define check_hip(function)                                                                        \
	{                                                                                              \
		auto err = function;                                                                       \
		check_hip_error(err, __FILE__, __LINE__);                                                  \
	}

#define force_hip(function)                                                                        \
	{                                                                                              \
		auto err = function;                                                                       \
		check_hip_error<true>(err, __FILE__, __LINE__);                                            \
	}

template<bool shouldThrow = false>
void check_hip_error(const hipError_t err, const char *filename, const int line)
{
	if(err != hipSuccess)
	{
		std::cout << "(" << err << ") in " << filename << " on line " << line << " : "
		          << hipGetErrorString(err) << std::endl;
		if constexpr(shouldThrow)
		{
			throw std::runtime_error(hipGetErrorString(err));
		}
	}
}

__global__ void simple_kernel()
{
	printf("Kernel cleared!\n");
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    hipStream_t stream;
    check_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    void* counting_buf;
    void* dst_buf;
    check_hip(hipExtMallocWithFlags(&counting_buf, hipMallocSignalMemory, sizeof(int)));
    check_hip(hipMalloc(&dst_buf, sizeof(int)));
    std::cout << "Rank " << my_rank << " - GPU Memory allocated" << std::endl;

    MPI_Win my_window;
    MPI_Group my_group;
    MPI_Win_create(counting_buf, buffer_size, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &my_window);
    std::cout << "Rank " << my_rank << " - Window created" << std::endl;

    check_hip(hipStreamWaitValue64(stream, counting_buf, size, hipStreamWaitValueEq, 0));
    simple_kernel<<<1, 1>>>();

    MPI_Win_free(&my_window);
    MPI_Group_free(&my_group);


    check_hip(hip_error = hipFree(counting_buf));
    check_hip(hip_error = hipFree(dst_buf));

    check_hip(hipStreamDestroy(stream));
    MPI_Finalize();
    return 0;
}