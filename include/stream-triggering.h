#include <stdint.h>
#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uintptr_t MPIX_ST_Queue;
typedef uintptr_t MPIX_ST_Queue_entry;

const int MPIX_SUCCESS = 0;
const int MPIX_ST_Queue_NULL = 0;
const int MPIX_ST_Queue_ENTRY_NULL = 0;

enum MPIX_ST_Queue_type
{
    THREAD = 0,
    THREAD_SERIALIZED = 1,
    CUDA = 2,
    AMD = 3,
    HPE = 4,
    MPICH_IMPL = 5
};

int MPIX_ST_Queue_host_wait(MPIX_ST_Queue);
int MPIX_ST_Queue_init(MPIX_ST_Queue*, MPIX_ST_Queue_type, void* extra_address);
int MPIX_ST_Queue_free(MPIX_ST_Queue*);
int MPIX_Prepare_all(int, MPI_Request[], MPIX_ST_Queue, MPIX_ST_Queue_entry[]);
int MPIX_Enqueue_entry(MPIX_ST_Queue, MPIX_ST_Queue_entry);
int MPIX_Enqueue_waitall(MPIX_ST_Queue);
int MPIX_ST_Queue_entry_free_all(int, MPIX_ST_Queue_entry[]);

#ifdef __cplusplus
}
#endif