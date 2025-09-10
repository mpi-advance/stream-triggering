#ifndef ST_ABSTRACT_QUEUE
#define ST_ABSTRACT_QUEUE

#include <stdint.h>

#include <memory>
#include <vector>

#include "request.hpp"

using namespace Communication;

/** @brief mostly virtual class to serve as interface across queue types
 *  @details
 *  Contains functions to be overridden based on the derived queue class
 *  \todo what to do about purely virtual functions?
*/
class Queue
{
public:
    virtual ~Queue() = default;
	
	/** @brief function to add requested operations to an internal list 
	 * @param [in, out] req Request object to add to pending operations queue. 
	 */
    virtual void enqueue_operation(std::shared_ptr<Request> req) = 0;
    
	/** @brief function to add requested operations to an internal list 
	 * @param [in, out] req Request object to add to pending operations queue. 
	 */
	virtual void enqueue_startall(
        std::vector<std::shared_ptr<Request>> reqs) = 0;
    
	/** @brief function to add requested operations to an internal list */
	virtual void enqueue_waitall()                  = 0;

	/** @brief function to wait on completion of operations by host.  */
    virtual void host_wait() = 0;

	/** @brief function match requests between devices 
	 * @param [in, out] request Request to match. 
	 */
    virtual void match(std::shared_ptr<Request> request);

	/** @brief conversion operator to allow Queue to be accessed using a uintptr_t **/
    operator uintptr_t() const
    {
        return (uintptr_t)(*this);
    }
};

#endif