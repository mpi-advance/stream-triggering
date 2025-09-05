/**
 * @file stream-triggering.h
*/

#ifndef MPI_ADVANCE_STREAM_TRIGGERING_H
#define MPI_ADVANCE_STREAM_TRIGGERING_H

#include <stdint.h>

#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uintptr_t                   MPIS_Queue; 
/**< wrapper around MPIS_Queue object so that it can be cast as necessary*/
typedef struct MPIS_Request_struct* MPIS_Request; 
/**< wrapper around MPIS_Request object so that it can be cast as necessary*/

/** \defgroup errors Function Return Codes 
* @{
*/
const int MPIS_SUCCESS               = 0;  /**<Function has completed successfully*/
const int MPIS_NOT_READY             = -1; /**<MPIS has not finished setup*/
const int MPIS_INVALID_REQUEST_STATE = -2; /**<Request has entered an invalid state*/
const int MPIS_UNSUPPORTED_BEHAVIOR  = -3; /**<Function has attempted to initiate unsupported behavior*/
const uintptr_t    MPIS_QUEUE_NULL   = 0;  /**<Check to see if queue exists*/
const MPIS_Request MPIS_REQUEST_NULL = 0;  /**<Check to see if request exists*/
/**@}*/

/**
	Supported MPIS_Queue Types based on underlying communication library
*/
enum MPIS_Queue_type
{
    THREAD            = 0, ///<Basic Threaded queue
    THREAD_SERIALIZED = 1, ///< 
    GPU_MEM_OPS       = 3, ///<Support for GPU memory
    HPE               = 4, ///<Support for HPE
    MPICH_IMPL        = 5, ///<MPICH optimized queue
    CXI               = 6  ///<CXI optimized queue
};

/**
 * Pointer to currently active queue (ONLY ONE QUEUE MAY BE ACTIVE AT A TIME)
*/
extern MPIS_Queue ACTIVE_QUEUE;


void MPIS_Hello_world();

// APIs from 7/25/24

/* Queue Management */
// int MPIS_Queue_fence();
/**
 * @brief This function deletes the supplied queue. Sets supplied pointer and ACTIVE_QUEUE to MPIS_QUEUE_NULL
 * @param [in, out] queue MPIS_Queue to be emptied and deleted
 * @return MPIS_Success upon completion
 */
int MPIS_Queue_free(MPIS_Queue*); 
/**
 * @brief This function creates a Queue with the supplied handle. Generated Queue type depends on the MPIS_Queue type supplied.  
 * @param [out] queue pointer to the generated queue object
 * @param [in] type MPIS_QUEUE_Type to be created. 
 * @param [in] pointer to back-end stream object. 
 * @return MPIS_Success upon completion, throws error if there is already an ACTIVE_QUEUE
 * /ref Queue, /ref MPIS_Queue_type
 */
int MPIS_Queue_init(MPIS_Queue*, MPIS_Queue_type, void*);
/**
 * @brief This function is a wrapper around the queue's host_wait function. 
 * @param [in] queue queue to wait on. 
 * @return MPIS_Success upon completion
 */
int MPIS_Queue_wait(MPIS_Queue);

/* Push stuff to Queue */
/**
 * @brief This function enqueues the request onto the streaming device 
 * @param [in] queue queue to start
 * @param [in] length the number of requests inside array of requests. 
 * @param [in] array_of_requests list of requests to enqueue. 
 * @return MPIS_Success upon completion
 */
int MPIS_Enqueue_startall(MPIS_Queue, int, MPIS_Request[]);
/**
 * @brief This function enqueues the request onto the streaming device 
 * @param [in] queue queue to start
 * @param [in] request Request to be enqueued
 * @return MPIS_Success upon completion
 */
int MPIS_Enqueue_start(MPIS_Queue, MPIS_Request*);
/**
 * @brief This function waits for the queue to finish all enqueue tasks
 * @param [in] queue queue to wait for completion
 * @return MPIS_Success upon completion
 */
int MPIS_Enqueue_waitall(MPIS_Queue);

/* New Matching Functions */
/**
 * @brief This function attempts to match 
 * @param [in] request request to match 
 * @param [in] status status object for the request.
 * @return MPIS_Success upon completion
 */
int MPIS_Match(MPIS_Request*, MPI_Status*);

/**
 * @brief This function synchronizes all the requests in the supplied list. 
 * @param [in] length number of requests in the supplied arrays. 
 * @param [in] requests array of requests to match
 * @param [in] status array of status objects for each of the requests. 
 * @return MPIS_Success upon completion
 */
int MPIS_Matchall(int, MPIS_Request[], MPI_Status[]);

/**
 * @brief This function synchronizes the supplied requests using one-sided communication 
 * @param [in] request 
 * @param [in] request 
 * @return MPIS_Success upon completion
 */
int MPIS_Imatch(MPIS_Request*, MPIS_Request*);

/**
 * @brief This function 
 * @param [in] request to the check status of
 * @param [out] result is set to 1 if true or 0 if false. 
 * @return MPIS_Success upon completion
 */
int MPIS_Is_matched(MPIS_Request*, int*);

/**
 * @brief This function matches the supplied request while it is in the queue???
 * @param [in] queue queue containing the request
 * @param [in, out] request Request to be matched 
 * @param [out] status the status object associated with the request
 * @return MPIS_Success upon completion
 */
int MPIS_Queue_match(MPIS_Queue, MPIS_Request*, MPI_Status*);

/*        Custom MPIS Override Functions         */
/* These have minimal, if any, new functionality */

/**
 * @brief This function deletes the supplied request
 * @param [out] request request to be freed and deleted. 
 * @return MPIS_Success upon completion
 */
int MPIS_Request_free(MPIS_Request*);

/**
 * @brief This function deletes each request in the supplied array
 * @param [in] length length of the supplied array
 * @param [int, out] requests array of request to be freed and deleted. 
 * @return MPIS_Success upon completion
 */
int MPIS_Request_freeall(int, MPIS_Request[]);

/**
 * @brief This function blocks until the supplied request completes
 * @param [in, out] request request to wait on 
 * @param [out] status the status of the supplied request
 * @return MPIS_Success upon completion
 */
int MPIS_Wait(MPIS_Request*, MPI_Status*);

/**
 * @brief This function blocks until the supplied requests complete
 * @param [in, out] requests array of request to wait on 
 * @param [out] status the status array of the supplied request
 * @return MPIS_Success upon completion
 */
int MPIS_Waitall(int, MPIS_Request[], MPI_Status[]);

/* Override Communication Functions */
int MPIS_Barrier_init(MPI_Comm, MPI_Info, MPIS_Request*);
// int MPIS_Bcast_init();

/**
 * @brief This function blocks until the supplied request completes
 * @param [in] buf buffer containing the message
 * @param [in] count number of elements in buf
 * @param [in] datatype the MPI_Datatype of the elements in buf 
 * @param [in] src the process rank sending the message
 * @param [in] tag tag to be used when matching messages
 * @param [in] comm MPI_Communicator to be used
 * @param [in] info MPI_info object to control behavior
 * @param [out] request request object generated for the request
 * @return MPIS_Success upon completion
 */
int MPIS_Recv_init(void*, MPI_Count, MPI_Datatype, int, int, MPI_Comm, MPI_Info,
       
/**
 * @brief This function blocks until the supplied request completes
 * @param [in] buf buffer containing the message
 * @param [in] count number of elements in buf
 * @param [in] datatype the MPI_Datatype of the elements in buf 
 * @param [in] dest the process to send the message
 * @param [in] tag tag to be used when matching messages
 * @param [in] comm MPI_Communicator to be used
 * @param [in] info MPI_info object to control behavior
 * @param [out] request request object generated for the request
 * @return MPIS_Success upon completion
 */	   MPIS_Request*);
int MPIS_Send_init(const void*, MPI_Count, MPI_Datatype, int, int, MPI_Comm,
                   MPI_Info, MPIS_Request*);


#ifdef __cplusplus
}
#endif
#endif