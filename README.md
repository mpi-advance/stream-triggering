# stream-triggering
An MPI Advance library exploring various stream triggering APIs

# Available backends
This implementation currently features four backends to use for the stream triggering APIs. The build of this library can have any number of them turned on at once, provided the proper libraries are on the system:
- **Thread**:  `-DUSE_THREAD_BACKEND=ON` - Uses a thread to offload communication too. Note that because this cannot tie to any GPU streams, the user must *manually* sync communication with the gpu. As such, this backend is purely for testing purposes and offers no real functionality over regular MPI.
- **HIP/CUDA**: `-DUSE_CUDA_BACKEND=ON` or `-DUSE_HIP_BACKEND=ON` - These two backends use the stream ops (`cuStreamWrite/WaitValue64` and `hipStreamWrite/WaitValue64`, respectively) to perform stream triggered communication. Note that the CUDA version may have issues running on some systems if the memory ops [are not enabled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html) (library may build, but not run.)
- **CXI**: `-DUSE_CXI_BACKEND` - This backend currently requires HIP support on the system to run, in addition to the requirement of the CXI libfabric provider. Uses extensions from the CXI libfabric provider to offer fully GPU driven communication.


## CXI Backend Limitations
Currently, the CXI backend needs to do a bidirectional data exchange in the match. To avoid _most_ collisions with other user MPI messages, and other stream_triggering communication matches, a separate communicator is made for each direction of the data exchange. However, if an application creates multiple communicators with overlapping ranks, and creates stream-triggered requests in a manner similar to this:
```c
MPI_Comm my_comm_a; // Custom communicator with at least ranks (0,1)
MPI_Comm my_comm_b; // Custom communicator with at least ranks (0,1)
MPIS_Request my_reqs[2];

// "rank 0"
MPIS_Send_init(/*to rank 1*/, my_comm_a, &my_reqs[0]); // A
MPIS_Send_init(/*to rank 1*/, my_comm_b, &my_reqs[1]); // B
// "rank 1"
MPIS_Recv_init(/*to rank 0*/, my_comm_b, &my_reqs[0]); // C
MPIS_Recv_init(/*to rank 0*/, my_comm_a, &my_reqs[1]); // D

MPIS_Matchall(2, my_reqs, MPI_STATUSES_IGNORE);

```
then a collision will likely still occur in the match. In this particular example, `A` will match with `C` instead of the user's desire to match `A` and `D`.
