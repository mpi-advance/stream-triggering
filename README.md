# stream-triggering
An MPI Advance library that offers a stream-triggering API with multiple implementations. The current set of APIs can be found the in the [main header](./include/stream-triggering.h), but full documentation of the semantics is currently only the paper below.

Regardless of the backend being used, all applications using our stream-triggered interface must call our `MPIS_` request creation function(s). This is because our approach also expects all requests to be "matched" before use. This is different from normal persistent MPI Requests, which will likely match every time they are started.

# Available Backends:
This implementation currently features four backends to use for the stream triggering APIs. The build of this library can have any number of them turned on at once, provided the proper libraries are on the system.

## Thread-based Backend:
Enabled with the CMake flag: `-DUSE_THREAD_BACKEND=ON`. This backend uses a thread to offload communication too. Note that because this cannot tie to any GPU streams, the user must *manually* sync communication with the GPU. As such, this backend is mostly for testing purposes. Technically, since requests are progressed in a different thread, this backend does offer slightly strong progress semantics than regular MPI, but otherwise this backend offers no new functionality over regular MPI.

## GPU Language-based Backend:
Enabled with the CMake flag `-DUSE_CUDA_BACKEND=ON` (for NVIDIA GPUs) or `-DUSE_HIP_BACKEND=ON` (for AMD GPUs). Both of these backends use the GPU stream memory operations (`cuStreamWrite/WaitValue64` and `hipStreamWrite/WaitValue64`, respectively) to perform stream triggered communication. Note that the CUDA version may have issues running on some systems if the memory ops [are not enabled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html) (library may build, but not run). These backends will still call the underlying MPI implementation for the desrired communication.

## CXI provider-based Backend:
Enabled with the CMake flag: `-DUSE_CXI_BACKEND`. This backend currently requires HIP support on the system to run, in addition to the requirement of the CXI libfabric provider. Uses extensions from the CXI libfabric provider to offer fully GPU driven communication. When building this mode, the GPU architecture for teh GPU must also be provided, as there are optimizations for certain GPUs.

# Paper
Current paper about our stream-triggering work:
```bibtex
@misc{bridges2026codesignevaluationcpufreempi,
      title={Co-Design and Evaluation of a CPU-Free MPI GPU Communication Abstraction and Implementation}, 
      author={Patrick G. Bridges and Derek Schafer and Jack Lange and James B. White III and Anthony Skjellum and Evan Suggs and Thomas Hines and Purushotham Bangalore and Matthew G. F. Dosanjh and Whit Schonbein},
      year={2026},
      eprint={2602.15356},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2602.15356}, 
}
```


# Acknowledgments
This work was also performed with support from the U.S. Department of Energy's National Nuclear Security Administration (NNSA) under the Predictive Science Academic Alliance Program (PSAAP-III), Award DE-NA0003966. 

Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. Department of Energy's National Nuclear Security Administration.
