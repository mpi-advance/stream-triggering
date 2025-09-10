#ifndef ST_CUDA_QUEUE
#define ST_CUDA_QUEUE

#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

/** @brief Derived class from QueueEntry to work with CUDA enabled GPU's
 * @details
 *	 includes overrides for the virtual functions in QueueEntry
 *   as well as flags and pointers for signaling between the cpu and gpu.
 */

class CudaQueueEntry : public QueueEntry
{
public:
    CudaQueueEntry(std::shared_ptr<Request> qe);
    ~CudaQueueEntry();
    /** @copydoc QueueEntry::start_host
	 *	@details 
	 *	
	 *		
	*/
    void start_host() override;
	
	  /** @copydoc QueueEntry::start_gpu
	 *	@details 
	 *	
	 *		
	*/
    void start_gpu(void*) override;
	
	/** @copydoc QueueEntry::wait_gpu
	 *	@details 
	 *	
	 *		
	*/
    void wait_gpu(void*) override;
	
    bool done() override;

protected:
	/** @brief location to signal when entry has started */ 
    int64_t* start_location;
	
	/** @brief location to signal when entry has finished */ 
    int64_t* wait_location;

	/** @brief device to start processing on */ 
    CUdeviceptr start_dev;
	
	/** @brief device to wait on*/ 
    CUdeviceptr wait_dev;
};

/** @brief Derived class from Queue to work with CudaStreams.
 * @details
 *	 includes overrides for the virtual functions in Queue
 *   contains pointer to Cudastream and monitoring thread. 
 */
class CudaQueue : public Queue
{
public
    /** @brief 
	 *	@details 
	 *		setups thread thr to run CudaQueue::progress()
	 * @param [in, out] stream cudaStream object to be leverage by this queue. 
	 */
    CudaQueue(cudaStream_t*);
	
	/** @brief sets shutdown switch for thread and waits for thread to finish. */
    ~CudaQueue();
    /** @copydoc Queue::enqueue_operation
	 *	@details 
	 *	
	 *		
	*/
    void enqueue_operation(std::shared_ptr<Request> req) override;
	
	/** @copydoc Queue::enqueue_startall
	 *	@details 
	 *	
	 *		
	*/
    void enqueue_startall(std::vector<std::shared_ptr<Request>> reqs) override;    
	
	/** @copydoc Queue::enqueue_waitall
	 *	@details 
	 *	
	 *		
	*/
    void enqueue_waitall() override;
	
	/** @copydoc Queue::host_wait
	 *	@details 
	 *	
	 *		
	*/
    void host_wait() override;

protected:
	/** @brief pointer to CudaStream executing requests */
    cudaStream_t* my_stream;

    /** @brief thread on cpu maintaining communication with GPU */
    std::thread thr;
	
	/** @brief boolean switch to kill stream once no longer needed.  */
    bool        shutdown = false;

    /** @brief mutex to lock thread when queue is being updated */
    std::mutex       queue_guard;
	
	/** @brief atomic counter for waiting, if >0 requests are still being processed */
    std::atomic<int> wait_cntr;

	/** @brief vector of requests to be processed */
    std::vector<std::reference_wrapper<QueueEntry>> entries;
	
	/** @brief vector of requests to start on the stream */
    std::vector<std::reference_wrapper<QueueEntry>> s_ongoing;
    
	/** @brief vector of requests currently being processed*/
	std::queue<std::reference_wrapper<QueueEntry>>  w_ongoing;

	/** @brief map of Queue Entries, keyed by request_id*/
    std::map<size_t, CudaQueueEntry> request_cache;

	/** @brief progress through the queue of Requests 
	 *  @details Goes through s_ongoing and starts each request
     *	         pushes QueueEnrty to w_ongoing
	 *           Goes through w_ongoing and checks if complete 
	 *           if so removes QueueEntry from w_ongoing.
	 *           Spins until shutdown flag tripped if no requests to process. 
	 */
    void progress();
};

#endif