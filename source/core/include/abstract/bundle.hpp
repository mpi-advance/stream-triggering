#ifndef ST_ABSTRACT_BUNDLE
#define ST_ABSTRACT_BUNDLE

#include <vector>

#include "entry.hpp"

namespace Communication
{

/** @brief This class is a bundle of QueueEntries to be processed by a thread.  
 *  @details 
 *		This class is focused around a vector of QueueEntry objects to
 *  	To be processed by a ThrQueue. Entries in the bundle can be 
 *		executed in serial or started in parallel. \n
 *      Note entries are NOT removed from item vector after completion
 * 
 */ 
	
class Bundle
{
public:
    /** @brief default constructor */
    Bundle() {};

	/** @brief add the supplied QueueEntry to internal items vector
	 *  @details 
	 *		moves QueueEntry onto the end of items
	 *      uses the QueueEntry move constructor @ref QueueEntry
	 *
	 *  @param [in] request QueueEntry to be added to items. 
	 */
    void add_to_bundle(QueueEntry& request)
    {
        items.push_back(request);
    }

	/** @brief starts each queued request serially in order of queuing
	 *  @details 
	 * 		Iterates through items starting each QueueEntry and waiting
	 *		for completion before starting next item. \n
	 *      Note entries are NOT removed from item vector after completion
	 *
	 */
    void progress_serial()
    {
        // Start and progress operations one at a time
        for (QueueEntry& req : items)
        {
            req.start_host();
            while (!req.done())
            {
                // Do nothing
            }
        }
    }
	
	/** @brief starts each queued request serially in order of queuing
	 *  @details 
	 * 		Starts all queued requests in items and waits for 
	 *		all of them to complete before returning. \n
	 *      Note entries are NOT removed from items vector after completion. 
	 *		
	 */
    void progress_all()
    {
        // Start all actions
        for (QueueEntry& req : items)
        {
            req.start_host();
        }

        // Wait for "starts" to complete:
        for (QueueEntry& req : items)
        {
            while (!req.done())
            {
                // Do nothing
            }
        }
    }

private:

`	/** @brief list of QueueEntries to process */
    std::vector<std::reference_wrapper<QueueEntry>> items;
};
}  // namespace Communication

#endif