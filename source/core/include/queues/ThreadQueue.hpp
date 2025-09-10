#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>

#include "abstract/bundle.hpp"
#include "abstract/entry.hpp"
#include "abstract/match.hpp"
#include "abstract/queue.hpp"

/**
 * @brief A thread based queue to process requests
 * @details  
 *   works with Bundles instead of QueueEntry
 * 
*/
template <bool isSerialized>
class ThreadQueue : public Queue
{
public:
    using InternalRequest = QueueEntry;
    using UserRequest     = std::shared_ptr<Request>;

	/** \todo why print out here?  */
    ThreadQueue() : thr(&ThreadQueue::progress, this)
    {
        Print::out("Thread Queue init-ed");
    }
	
	/** @brief kills thread when object is deleted;''
	*/
    ~ThreadQueue()
    {
        shutdown = true;
        thr.join();
    }

	/** @brief adds request to bundle */  
    void enqueue_operation(UserRequest request) override
    {
        size_t request_id = request->getID();
        if (!request_cache.contains(request_id))
        {
            // Also converts to InternalRequest
            request_cache.emplace(request_id, request);
        }
        entries.add_to_bundle(request_cache.at(request_id));
    }

    void enqueue_startall(std::vector<UserRequest> requests) override
    {
        for (auto& req : requests)
        {
            enqueue_operation(req);
        }
    }

	/** @brief start bundle??
	 * @details
	 *  Moves Bundle to work queue
	 *  Add 1 instance to busy counter. 
	 * 
	 */
    void enqueue_waitall() override
    {
        std::scoped_lock<std::mutex> incoming_lock(queue_guard);
        // Move Bundle (entries) to the queue of work
        pending.push(std::move(entries));
        // Add one to busy counter
        busy += 1;
        // Remake entries
        entries = Bundle();
    }

	/** @brief  does nothing as there is no external device to wait for. */
    void host_wait() override
    {
        while (busy.load())
        {
            // Do nothing.
        }
    }

protected:
    // Thread control variables
	/** @brief flag, if true thread has/is doing work */
    std::atomic<int> busy; 
    
	/** @brief handle to thread */
	std::thread      thr;  
	
	/** @brief boolean flag to signal when to kill thread */
    bool             shutdown = false; 
    
	/** @brief mutex to lock queue from editing when modifying from the queue*/
	std::mutex       queue_guard;     

    // Bundle variables
    using BundleIterator = std::vector<Bundle>::iterator; //<
	/**@brief  Bundle of requests to process */
    Bundle                            entries;            //< 
    /**@brief bundles that are currently are being processed.  */
	std::queue<Bundle>                pending;            //< 
    /**@brief  requests that have been added to queue, keyed by request_id */
	std::map<size_t, InternalRequest> request_cache;      //< List of requests 

	/** @brief Function to progress requests included in the bundle
	 *	@details 
	 *  This function spins until the busy flag is set. Once the busy flag is set
	 *  One the thread has work, it begins processing the first bundle
	 *  If serialized, calls progress_serial() \n
	 *  If not serialized, start all bundles \n
	 *  Inter loop returns when all active bundles are finished. \n
	 *  thread ends and rejoins when shutdown flag is tripped. 
	 *
	 */ 
    void progress()
    {
        while (!shutdown)
        {
            if (busy > 0)
            {
                Bundle the_bundle = std::move(pending.front());
                {  // Scope of the lock
                    std::scoped_lock<std::mutex> incoming_lock(queue_guard);
                    pending.pop();
                }

                if constexpr (isSerialized)
                {
                    the_bundle.progress_serial();
                }
                else
                {
                    the_bundle.progress_all();
                }
                busy--;
            }
            else
            {
                std::this_thread::yield();
            }
        }
    }
};

#endif