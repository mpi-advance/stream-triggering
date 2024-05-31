#include <stdint.h>
#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uintptr_t MPIX_Queue;
typedef uintptr_t MPIX_Queue_entry;

const int MPIX_SUCCESS = 0;
const int MPIX_QUEUE_NULL = 0;
const int MPIX_QUEUE_ENTRY_NULL = 0;

enum MPIX_Queue_type
{
    THREAD = 0,
    THREAD_SERIALIZED = 1,
    CUDA = 2,
    AMD = 3,
    HPE = 4,
    MPICH = 5
};

int MPIX_Queue_host_wait(MPIX_Queue);
int MPIX_Queue_init(MPIX_Queue*, MPIX_Queue_type);
int MPIX_Queue_free(MPIX_Queue*);
int MPIX_Prepare_all(int, MPI_Request[], MPIX_Queue, MPIX_Queue_entry[]);
int MPIX_Enqueue_entry(MPIX_Queue, MPIX_Queue_entry);
int MPIX_Enqueue_waitall(MPIX_Queue);
int MPIX_Queue_entry_free_all(int, MPIX_Queue_entry[]);

#ifdef __cplusplus
}
#endif