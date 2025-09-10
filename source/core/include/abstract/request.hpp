/** @file request.hpp
*/

#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include <vector>
#include <string.h>

#include "misc/print.hpp"
#include "safety/mpi.hpp"

namespace Communication
{
\
/** @brief Supported MPI Operations
 *
 *
 */
enum Operation
{
    SEND,   //!< Send a message to a remote process */
    RECV,   //!< Recieve a message from a remote process */
    BARRIER //!< Synchronize progress among processes */
};

/** @brief SUPPORTED GPU MEMORY TYPES */
enum GPUMemoryType
{
    COARSE = 1, //!< Memory coherency enforced only at kernel boundaries*/
    FINE   = 2, //!< Additional synchronization to maintain Memory coherency during runtime */
};

/** @brief Contains information and controls for the required MPI operation. 
 * @details
 * This class contains information for a single mpi operation to be 
 * run in a queue. 
 *
 */
class Request
{
public:
	/** @brief operation to be conducted.  */
    Operation    operation; 
	/** @brief buffer containing data to be sent.  */
    void*        buffer;
    /** @brief number of elements in the buffer to be sent.  */
	MPI_Count    count;
    /** @brief datatype of the elements in the buffer.  */
	MPI_Datatype datatype;
    /** @brief rank of remote process to communicate with.  */
	int          peer;
    /** @brief tag to aid in accurate message matching  */
	int          tag;
    /** @brief context for rank and process addressing  */
	MPI_Comm     comm;
    /** @brief MPI_Info object containing control values for the operation.  
	 * @details
	 * The MPI_Info object uses the following key/value pairs 
	 * Key
	 *
	 */
	MPI_Info     info;

	/** @brief Basic constructor for the object.
	 * @details 
	 * Requires all information necessary to setup a normal MPI communicatin channel. 
	 * Also requires the operation to be specifically called out as the actual operation will be 
	 * set up at a latter time. Is used as core of @ref QueueEntry object. 	
	 */
    Request(Operation _operation, void* _buffer, MPI_Count _count, MPI_Datatype _datatype,
            int _peer, int _tag, MPI_Comm _comm, MPI_Info _info)
        : operation(_operation),
          buffer(_buffer),
          count(_count),
          datatype(_datatype),
          peer(_peer),
          tag(_tag),
          comm(_comm),
          info(_info),
          myID(assignID()),
          matched(false)
    {
        int size = -1;
        check_mpi(MPI_Type_size(_datatype, &size));
        Print::out("Request made with address, size, count, tag, and ID:", _buffer, size,
                   _count, tag, myID);

        constexpr int string_size = 10;
        char          info_key[]  = "MPIS_GPU_MEM_TYPE";
        char          value[string_size];
        int           flag = 0;
        // Pre MPI-4.0
        if (MPI_INFO_NULL != _info)
        {
            force_mpi(MPI_Info_get(_info, info_key, string_size, value, &flag));
        }

        if (0 == strcmp(value, "FINE"))
        {
            Print::out("Using fine-grained memory!");
            memory_type = GPUMemoryType::FINE;
        }
        else
        {
            Print::out("Using coarse-grained memory!");
            memory_type = GPUMemoryType::COARSE;
        }
    };

	/** @brief checks if request has been matched
	 * @return true if matched, false otherwise. 
	*/
    bool is_matched()
    {
        return matched;
    }

	/** @brief gets id of this Request
	 * @return id of request
	*/
    size_t getID()
    {
        return myID;
    }
 
    /** @brief Function sets vector of requests to match against
	 * @details 
	 * Expands match_requests to be able to hold num MPI_Requests.
     *
	 * @param [in] number of requests to create in vector
	 * @return pointer to first MPI_Request inside match_requests
	 */
    MPI_Request* get_match_requests(size_t num)
    {
        match_requests = std::vector<MPI_Request>(num, MPI_REQUEST_NULL);
        match_statuses = std::vector<MPI_Status>(num);
        return match_requests.data();
    }

    /** @brief Function sets vector of requests to match against
	 * @details 
	 * Expands match_requests to be able to hold num MPI_Requests.
     *
	 * @param [in] number of requests to create in vector
	 * @return pointer to first MPI_Request inside match_requests
	 */
    void wait_on_match()
    {
        check_mpi(MPI_Waitall(match_requests.size(), match_requests.data(),
                              match_statuses.data()));
        matched = true;
    }

	/** @brief returns the GPUMemoryType associated with the request */
    GPUMemoryType get_memory_type()
    {
        return memory_type;
    }

protected: 
	/** @brief unique ID for the request*/
    size_t                   myID;
	/** @brief Memory type to be used by the Request*/
    GPUMemoryType            memory_type;
	/** @brief list of requests to match */
    std::vector<MPI_Request> match_requests;
	/** @brief list of status of requests matched  */
    std::vector<MPI_Status>  match_statuses;
	
	/** @brief Has request been matched*/
    bool                     matched = false;
	
	/** @brief Function to assign unique ID to each Request*/
    static size_t assignID()
    {
        static size_t ID = 1;
        return ID++;
    }
};

}  // namespace Communication
#endif