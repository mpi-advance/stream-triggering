#ifndef ST_ABSTRACT_BUNDLE
#define ST_ABSTRACT_BUNDLE

#include <vector>

#include "entry.hpp"

namespace Communication
{
class Bundle
{
public:
    // No fancy constructor
    Bundle() {};

	/**
	 * add queue_entry to TO DO vector
	 */
    void add_to_bundle(QueueEntry& request)
    {
        items.push_back(request);
    }

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
    std::vector<std::reference_wrapper<QueueEntry>> items;
};
}  // namespace Communication

#endif