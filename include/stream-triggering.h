#include <stdint.h>

#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uintptr_t MPIS_Queue;
typedef uintptr_t MPIS_Request;

/* Errors */
const int MPIS_SUCCESS   = 0;
const int MPIX_NOT_READY = -1;

const uintptr_t MPIS_QUEUE_NULL   = 0;
const uintptr_t MPIS_REQUEST_NULL = 0;

enum MPIS_Queue_type
{
    THREAD            = 0,
    THREAD_SERIALIZED = 1,
    GPU_MEM_OPS       = 3,
    HPE               = 4,
    MPICH_IMPL        = 5,
    CXI               = 6
};

extern MPIS_Queue ACTIVE_QUEUE;

// APIs from 7/25/24

/* Queue Management */
// int MPIS_Queue_fence();
int MPIS_Queue_free(MPIS_Queue*);
int MPIS_Queue_init(MPIS_Queue*, MPIS_Queue_type, void*);
int MPIS_Queue_wait(MPIS_Queue);

/* Custom Communication Methods */
int MPIS_Barrier_init(MPI_Comm, MPI_Info, MPIS_Request*);
// int MPIS_Bcast_init();
int MPIS_Recv_init(void*, MPI_Count, MPI_Datatype, int, int, MPI_Comm, MPI_Info,
                   MPIS_Request*);
int MPIS_Send_init(const void*, MPI_Count, MPI_Datatype, int, int, MPI_Comm,
                   MPI_Info, MPIS_Request*);

/* Push stuff to Queue */
int MPIS_Enqueue_startall(MPIS_Queue, int, MPIS_Request[]);
int MPIS_Enqueue_start(MPIS_Queue, MPIS_Request);
int MPIS_Enqueue_waitall(MPIS_Queue);

/* New Matching Functions */
int MPIS_Match(MPIS_Request);
// int MPIS_Matchall();
// int MPIS_Imatch();
int MPIS_Is_matched(MPIS_Request, int*);
int MPIS_Queue_match(MPIS_Queue, MPIS_Request);

/*        Custom MPIS Override Functions         */
/* These have minimal, if any, new functionality */
int MPIS_Request_free(MPIS_Request*);
int MPIS_Request_freeall(int, MPIS_Request[]);
// int MPIS_Wait();

// End APIs from 7/25/24

#ifdef __cplusplus
}
#endif