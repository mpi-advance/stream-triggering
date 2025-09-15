#ifndef ST_HIP_QUEUE
#define ST_HIP_QUEUE

#include <hip/hip_runtime.h>

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

#include "abstract/entry.hpp"
#include "abstract/queue.hpp"

/** \defgroup HIP HIP Backend
 * @brief Internal Functions utilized when the HIP Backend is called. 
 * @ingroup backends
 */

/** @brief Derived class from QueueEntry to work with HIP enabled GPU's
 * @details
 *	 includes overrides for the virtual functions in QueueEntry
 *   as well as flags and pointers for signaling between the cpu and gpu.
 *  @ingroup HIP
 */

class HIPQueueEntry : public QueueEntry
{
public:
    HIPQueueEntry(std::shared_ptr<Request> qe);
    ~HIPQueueEntry();

    void start_host() override;
    void start_gpu(void*) override;
    void wait_gpu(void*) override;
    bool done() override;

protected:
	/** @brief location to signal when entry has started */ 
	int64_t* start_location;
	
	/** @brief location to signal when entry has finished */ 
    int64_t* wait_location;

    void* start_dev;
    void* wait_dev;
};

/** @brief Derived class from Queue to work with HIP
 * @details
 *	 includes overrides for the virtual functions in Queue
 *   contains pointer to Cudastream and monitoring thread. 
 *  @ingroup HIP
 */
class HIPQueue : public Queue
{
public:
    HIPQueue(hipStream_t*);
    ~HIPQueue();

    void enqueue_operation(std::shared_ptr<Request> req) override;
    void enqueue_startall(std::vector<std::shared_ptr<Request>> reqs) override;
    void enqueue_waitall() override;
    void host_wait() override;

protected:
	/** @brief pointer to hipStream executing requests */
    hipStream_t* my_stream;
	
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
    
	/** @brief map of Queue Entries, keyed by request_id*/
	std::queue<std::reference_wrapper<QueueEntry>>  w_ongoing;

	/** @brief progress through the queue of Requests 
	 *  @details Goes through s_ongoing and starts each request
     *	         pushes QueueEnrty to w_ongoing
	 *           Goes through w_ongoing and checks if complete 
	 *           if so removes QueueEntry from w_ongoing.
	 *           Spins until shutdown flag tripped if no requests to process. 
	 */
    std::map<size_t, HIPQueueEntry> request_cache;

    void progress();
};

#endif