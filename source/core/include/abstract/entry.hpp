#ifndef ST_ABSTRACT_QUEUE_ENTRY
#define ST_ABSTRACT_QUEUE_ENTRY

#include <memory>

#include "request.hpp"

using namespace Communication;

// Common class for MPI-based backends (Thread, HIP, CUDA)

/** @brief Wrapper class around MPI_request for execution by Queue
 * @details 
 *	 Base MPI_request is protected in mpi_request. 
 *	 A std::shared_ptr to the request is used to allow access to the request. 
 * 	 Virtual Functions can be overwrote by code for specific Queue_types being used. \n
 *	 Functions in the class control the creation, and movement of the object 
 *   Only one copy of each instance of this object should exist at a time. 
 *
*/
class QueueEntry
{
public:

	/** @brief Constructor for QueueEntry calls MPI function based on operation in req. 
	 * @details 
	 *	 Selects and executes a MPI Function call based on the requested operation.
	 *   If the MPI Function outputs a mpi_request object, that object is stored as the 
	 *   the internal mpi_request for the QueueEntry object. 
	 *   If barrier operation no request is generated. 
     *
	 * @param [in] req Request to interpret core operation from 
	 * \todo why explicit?
	*/
    explicit QueueEntry(std::shared_ptr<Request> req) : original_request(req), threshold(0)
    {
        switch (req->operation)
        {
            case Communication::Operation::SEND:
                check_mpi(MPI_Send_init(req->buffer, req->count, req->datatype,
                                        req->peer, req->tag, req->comm,
                                        &mpi_request));
                break;
            case Communication::Operation::RECV:
                check_mpi(MPI_Recv_init(req->buffer, req->count, req->datatype,
                                        req->peer, req->tag, req->comm,
                                        &mpi_request));
                break;
            case Communication::Operation::BARRIER:
                break;
            default:
                throw std::runtime_error("Invalid Request");
                break;
        }
    }

	/** @brief Virtual Deconstructor for the object. 
	 *  @details 
	 *  Should be overridden by the active Queue type.
     *  Checks if mpi_request exists before attempting to free it. 
	 * 
	 */
    virtual ~QueueEntry()
    {
        if (MPI_REQUEST_NULL != mpi_request && original_request &&
            original_request->operation != Communication::Operation::BARRIER)
        {
            check_mpi(MPI_Request_free(&mpi_request));
        }
    }

    /** @brief disable the default copy constructor **/
    QueueEntry(const QueueEntry& other)            = delete;
	/** @brief disables the assignment operator **/
    QueueEntry& operator=(const QueueEntry& other) = delete;

    /** @brief move constructor 
	 *  @details 
	 *  	Sets pointers and objects in current object. 
 	 *      Deletes pointer and reference in old object. 
	 *
	 * @param [in, out] other the other QueueEntry to read and reset. 
	 */
    QueueEntry(QueueEntry&& other) noexcept
        : mpi_request(other.mpi_request),
          original_request(other.original_request)
    {
        // clear other structs
        other.mpi_request = MPI_REQUEST_NULL;
        other.original_request.reset();
    }
    
	 /** @brief assignment move operator 
	 *  @details 
	 *  	Sets pointers and objects in current object. 
 	 *      Deletes pointer and reference in old object. 
	 *
	 * @param [in, out] other the other QueueEntry to read and reset. 
	 */
	QueueEntry& operator=(QueueEntry&& other) noexcept
    {
        if (this != &other)
        {
            mpi_request      = other.mpi_request;
            original_request = other.original_request;
            // clear other
            other.mpi_request = MPI_REQUEST_NULL;
            other.original_request.reset();
        }
        return *this;
    }

	/** @brief call to start the operation operation on the host. 
	 * @details 
	 *	Should be overridden by active Queue type to match host device
	 *  If not overridden calls MPI_Ibarrier if the queued operation is a barrier
	 *  or MPI_Start on the internal request. 
	*/
    virtual void start_host()
    {
        if (original_request->operation == Communication::Operation::BARRIER)
        {
            check_mpi(MPI_Ibarrier(original_request->comm, &mpi_request));
        }
        else
        {
            check_mpi(MPI_Start(&mpi_request));
        }
    }

	/** @brief Virtual function to start gpu
	 *  @details
	 *  Should be overloaded if stream is going to be run on GPU
	 *  Function when overridden should start supplied stream. 
     *  If not overridden (or gpu is not used) does nothing. 
	 *  @param [in, out] stream stream to be started on the gpu
	 */
    virtual void start_gpu(void* stream)
    {
        // Does nothing in base class.
    }

	/** @brief Virtual function to start stream
	 *  @details
	 *  Should be overloaded if stream is going to be run on GPU
	 *  Function when overridden should start supplied stream. 
     *  If not overridden, increments threshold flag, and starts host and starts_gpu.
     *  The called functions should be overridden to match the structure of the host. 	 
	 *  @param [in, out] stream stream to be started. 
	 */
    virtual void start(void* stream)
    {
        threshold++;
        start_host();
        start_gpu(stream);
    }

	/** @brief Virtual function to wait on completion of a stream 
	 *  @details
	 *  Should be overloaded if stream is going to be run on GPU
	 *  Function when overridden should start supplied stream. 
     *  If not overridden (e.g no host) does nothing. 
	 *  @param [in, out] stream stream to wait on. 
	 */
    virtual void wait_gpu(void* stream)
    {
        // Does nothing in base class.
    }

	/** @brief Wrapper around MPI_Test to see if internal request is complete. 
	*   @returns 1 if complete, 0 otherwise. 
	*/
    virtual bool done()
    {
        int value = 0;
        check_mpi(MPI_Test(&mpi_request, &value, MPI_STATUS_IGNORE));
        return value;
    }

protected:
    /** @brief variable to act as signal flag between host device and GPU. */
    int64_t threshold = 0;

	/** @brief internal MPI_Request to be used to monitor operation status */
    MPI_Request              mpi_request      = MPI_REQUEST_NULL; 
    
	/** @brief shared pointer to internal MPI_Request */
	std::shared_ptr<Request> original_request = nullptr;          
};

#endif