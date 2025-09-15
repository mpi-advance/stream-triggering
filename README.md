# stream-triggering
An MPI Advance library exploring various stream triggering APIs

# Overview
This library supports triggering basic MPI operations using different streaming methods. 
Currently supported streams include: Threads, Cudastreams, HIP streams, and CXI implemented libfabric streams

# Building Instructions

## Prerequisites
Inorder to build the library the following packages are required
- MPI implementation supporting MPI 3.0+
- CMake 3.17+
- C23+

Additional packages may be required based on the selected backend. 



## Instructions
The stream-triggering library is a fairly simple CMake build:
```bash
mkdir <build_dir>
cd <build_dir>
cmake <options> ..
make <options>
```

### CMake Build Options
- `-USE_THREAD_BACKEND` (ON) : Use thread-based backend
- `-USE_HIP_BACKEND` (OFF) : Use HIP backend
- `-USE_CUDA_BACKEND` (OFF) : Use Cuda backend, requires CUDA 17+
- `-USE_CXI_BACKEND` (OFF) :  Use HPE CXI Libfabric provider 

### Accessing the Library
In order to use the library, you will need to make sure it is either included in RPATH or the containing directory is added to LD_LIBRARY_PATH and you will need to include the supplied `stream-triggering.h`.  

# Basic Library Operations
The library requires a basic ordering of function calls to work as designed. 
There can be only one active Queue per process. 

1. The stream must be created. (unless using thread backend) 
then the queue must be initialized.
Requests must be matched
 
Then one or more requests can be placed in the queue using the enqueue function. 

1. MPIS_Queue_init
2. MPIS_Match
2. MPIS_Eneque_start 

4. MPIS_Enqueue_waitall
MPIS_Queue_wait
MPIS_Queue_free


## Stream-Triggering Library APIs
```c
MPIS_Queue_init()
MPIS_Send_init()
MPIS_Recv_init()
MPIS_Match()
MPIS_Enqueue_start()
MPIS_Enqueue_startall()
MPIS_Enqueue_waitall()
MPIS_Queue_wait()
MPIS_Queue_free()
```
### Minimally Changed MPI functions
```c
int MPIS_Request_free(MPIS_Request*);
int MPIS_Request_freeall(int, MPIS_Request[]);
int MPIS_Wait(MPIS_Request*, MPI_Status*);
int MPIS_Waitall(int, MPIS_Request[], MPI_Status[]);
int MPIS_Barrier_init(MPI_Comm, MPI_Info, MPIS_Request*);
```

## Library Structs
Information about classes, structures, and internal functions may be accessed by using Doxygen with the supplied .Doxyfile (run `doxygen .Doxyfile` from the top level of this repo).

### Acknowledgments
This work has been partially funded by ...

 

 