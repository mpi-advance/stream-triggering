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

/** \defgroup user_api User-facing MPIS functions
 * @brief Functions designed to be called by end users
 * @details 
 *  
 */


/** @brief wrapper that can point to any of the several derived classes of MPIS_Queue. */
typedef uintptr_t                   MPIS_Queue; 

/** @brief pointer to MPIS_Request object or one of its derived classes. */
typedef struct MPIS_Request_struct* MPIS_Request; 

/** \defgroup errors MPIS Function Return Codes
 * @brief Enumerated error codes for debugging the library.
 * @details 
 *  
 * @{
 */
const int MPIS_SUCCESS               = 0;  //!< Function has completed successfully*/
const int MPIS_NOT_READY             = -1; //!< MPIS has not finished setup*/
const int MPIS_INVALID_REQUEST_STATE = -2; //!< Request has entered an invalid state*/
const int MPIS_UNSUPPORTED_BEHAVIOR  = -3; //!< Function has attempted to initiate unsupported behavior*/
const uintptr_t    MPIS_QUEUE_NULL   = 0;  //!< Check to see if queue exists*/
const MPIS_Request MPIS_REQUEST_NULL = 0;  //!< Check to see if request exists*/
/**@}*/

/** @brief Supported MPIS_Queue Types based on underlying communication library
 *	@details 
 *	
 */
enum MPIS_Queue_type
{
    THREAD            = 0, ///<Basic Threaded queue
    THREAD_SERIALIZED = 1, ///< Serialized Thread queue
    GPU_MEM_OPS       = 3, ///<Support for GPU memory
    HPE               = 4, ///<Support for HPE
    MPICH_IMPL        = 5, ///<MPICH optimized queue
    CXI               = 6  ///<CXI optimized queue
};

/** @brief Pointer to currently active queue (ONLY ONE QUEUE MAY BE ACTIVE AT A TIME)
 * @details
 * 
 * 
 */
extern MPIS_Queue ACTIVE_QUEUE;

/** @brief Function to debug and test MPI_Init issues
 * @details 
 */
void MPIS_Hello_world();

// APIs from 7/25/24

/* Queue Management */
// int MPIS_Queue_fence();
/* @brief 
 * @details  
 *
 * @ingroup user_api
 * @param 
 * @return 
 */
int MPIS_Queue_free(MPIS_Queue*); 
/** @brief This function creates a Queue with the supplied handle. 
 * @details 
 * The generated Queue type depends on the MPIS_Queue type supplied to the function.   
 *
 * @ingroup user_api
 * @param [out] queue pointer to the generated queue object
 * @param [in] type MPIS_QUEUE_Type to be created. 
 * @param [in] pointer to back-end stream object. 
 * @return MPIS_Success upon completion, throws error if there is already an ACTIVE_QUEUE
 * /ref Queue, /ref MPIS_Queue_type
 */
int MPIS_Queue_init(MPIS_Queue*, MPIS_Queue_type, void*);
/**
 * @brief This function is a wrapper around the queue's host_wait function. 
 * @ingroup user_api
 * @param [in] queue queue to wait on. 
 * @return MPIS_Success upon completion
 */
int MPIS_Queue_wait(MPIS_Queue);

/* Push stuff to Queue */
/**
 * @brief This function enqueues the request onto the streaming device 
 * @details
 * 
 * @ingroup user_api
 * @param [in] queue queue to start
 * @param [in] length the number of requests inside array of requests. 
 * @param [in] array_of_requests list of requests to enqueue. 
 * @return MPIS_Success upon completion
 */
int MPIS_Enqueue_startall(MPIS_Queue, int, MPIS_Request[]);
/**
 * @brief This function enqueues the request onto the streaming device 
 * @details
 *
 * @ingroup user_api
 * @param [in] queue queue to start
 * @param [in] request Request to be enqueued
 * @return MPIS_Success upon completion
 */
int MPIS_Enqueue_start(MPIS_Queue, MPIS_Request*);
/**
 * @brief This function waits for the queue to finish all enqueue tasks
 * @details
 *
 * @ingroup user_api
 * @param [in] queue queue to wait for completion
 * @return MPIS_Success upon completion
 */
int MPIS_Enqueue_waitall(MPIS_Queue);

/* New Matching Functions */
/**
 * @brief 
 * @details
 *	Wrapper around queue->match function. @ref Queue::match 
 * 
 * @ingroup user_api
 * @param [in] request request to match 
 * @param [in] status status object for the request.
 * @return MPIS_Success upon completion
 */
int MPIS_Match(MPIS_Request*, MPI_Status*);

/**
 * @brief This function synchronizes all the requests in the supplied list. 
 * @details
 *
 * @ingroup user_api
 * @param [in] length number of requests in the supplied arrays. 
 * @param [in] requests array of requests to match
 * @param [in] status array of status objects for each of the requests. 
 * @return MPIS_Success upon completion
 */
int MPIS_Matchall(int, MPIS_Request[], MPI_Status[]);

/**
 * @brief This function synchronizes the supplied requests using one-sided communication 
 * @details
 *
 * @param [in] request 
 * @param [in] request 
 * @return MPIS_Success upon completion
 */
int MPIS_Imatch(MPIS_Request*, MPIS_Request*);

/**
 * @brief This function checks if the supplied request has been matched or not. 
 * @details
 * 
 * @param [in] request to the check status of
 * @param [out] result is set to 1 if true or 0 if false. 
 * @return MPIS_Success upon completion
 */
int MPIS_Is_matched(MPIS_Request*, int*);

/**
 * @brief This function matches the supplied request while it is in the queue???
 * @details
 *
 * @param [in] queue queue containing the request
 * @param [in, out] request Request to be matched 
 * @param [out] status the status object associated with the request
 * @return MPIS_Success upon completion
 */
int MPIS_Queue_match(MPIS_Queue, MPIS_Request*, MPI_Status*);

/*        Custom MPIS Override Functions         */
/* These have minimal, if any, new functionality */

/** @defgroup MPI_Overrides MPI Overrides
 * @brief MPI functions with minimum alternation to function with the library. 
 * @details These functions have been minimally modified from the MPI Standard
 * to work with the MPIS_Request object. For more detail about the behavior and
 * use of these functions please refer to the MPI_Standard. 
 * @{
 */
int MPIS_Request_free(MPIS_Request*);
int MPIS_Request_freeall(int, MPIS_Request[]);
int MPIS_Wait(MPIS_Request*, MPI_Status*);
int MPIS_Waitall(int, MPIS_Request[], MPI_Status[]);
int MPIS_Barrier_init(MPI_Comm, MPI_Info, MPIS_Request*);
/** @} */


// int MPIS_Bcast_init();

/**
 * @brief This function blocks until the supplied request completes
 * @details
 *
 * * @ingroup user_api
 * @param [in] buf buffer containing the message
 * @param [in] count number of elements in buf
 * @param [in] datatype the MPI_Datatype of the elements in buf 
 * @param [in] src the process rank sending the message
 * @param [in] tag tag to be used when matching messages
 * @param [in] comm MPI_Communicator to be used
 * @param [in] info MPI_info object to control behavior
 * @param [out] request request object generated for the request
 * @return MPIS_Success upon completion
 
 \todo confirm where MPI_Info object options/details need to go. 

 
 */
int MPIS_Recv_init(void*, MPI_Count, MPI_Datatype, int, int, MPI_Comm, MPI_Info,
					MPIS_Request*);
       
/**
 * @brief This function blocks until the supplied request completes
 * @details
 *
 * @ingroup user_api
 * @param [in] buf buffer containing the message
 * @param [in] count number of elements in buf
 * @param [in] datatype the MPI_Datatype of the elements in buf 
 * @param [in] dest the process to send the message
 * @param [in] dest the process to send the message
 * @param [in] tag tag to be used when matching messages
 * @param [in] comm MPI_Communicator to be used
 * @param [in] info MPI_info object to control behavior
 * @param [out] request request object generated for the request
 * @return MPIS_Success upon completion
 */	
int MPIS_Send_init(const void*, MPI_Count, MPI_Datatype, int, int, MPI_Comm,
                   MPI_Info, MPIS_Request*);


#ifdef __cplusplus
}
#endif
#endif