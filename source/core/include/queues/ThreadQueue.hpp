#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include "abstract/queue.hpp"

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>

class ThreadQueueEntry : public QueueEntry
{
public:
	ThreadQueueEntry(MPI_Request req) : QueueEntry(req) {}
	~ThreadQueueEntry() = default;

	void prepare() override;

	void start() override;

	bool done() override;

	void progress() override;
};

template<bool isSerialized>
class ThreadQueue : public Queue
{
public:
	ThreadQueue() : thr(&ThreadQueue::progress, this) {}
	~ThreadQueue()
	{
		shutdown = true;
		thr.join();
	}

	QueueEntry *create_entry(MPI_Request req) override
	{
		return new ThreadQueueEntry(req);
	}

	void enqueue_operation(QueueEntry *qe) override
	{
		std::scoped_lock<std::mutex> incoming_lock(queue_guard);
		entries.push_back(qe);
	}

	void enqueue_waitall() override
	{
		std::scoped_lock<std::mutex> incoming_lock(queue_guard);
		size_t                       amt = entries.size();
		stop_counts.push(amt);
		pending.insert(pending.begin(), entries.begin(), entries.end());
		busy += amt;
		entries.clear();
	}

	void host_wait() override
	{
		while(busy.load())
		{
			// Do nothing.
			// No lock because this function doesn't want to set
			// amount_to_do to any value, only read.
		}
	}

protected:
	std::atomic<int> busy;
	std::thread      thr;
	bool             shutdown = false;

	std::mutex queue_guard;

	std::queue<size_t>        stop_counts;
	std::vector<QueueEntry *> pending;
	std::vector<QueueEntry *> ongoing;

	void progress()
	{
		// Thread specific variables
		size_t amount_to_do = 0;

		while(!shutdown)
		{
			if(amount_to_do == 0 && stop_counts.size() > 0)
			{
				// Determine how much we need to do
				{ // Scope of the lock
					std::scoped_lock<std::mutex> incoming_lock(queue_guard);
					amount_to_do = stop_counts.front();
					stop_counts.pop();
					ongoing.insert(
					    ongoing.begin(), pending.begin(), pending.begin() + amount_to_do);
					pending.erase(pending.begin(), pending.begin() + amount_to_do);
				}

				if constexpr(isSerialized)
				{
					progress_options_serial(amount_to_do);
				}
				else
				{
					progress_options_all(amount_to_do);
				}
			}
			else
			{
				std::this_thread::yield();
			}
		}
	}

	void progress_options_all(size_t &amount_to_do)
	{
		// Start operations
		for(std::vector<QueueEntry *>::iterator entry = ongoing.begin(); entry != ongoing.end();
		    entry++)
		{
			(*entry)->start();
		}

		// Progress them and watch out for shutdown (just in case)
		while(amount_to_do != 0 && !shutdown)
		{
			for(std::vector<QueueEntry *>::iterator entry = ongoing.begin(); entry != ongoing.end();
			    entry++)
			{
				(*entry)->progress();
				if((*entry)->done())
				{
					ongoing.erase(entry);
					amount_to_do--;
					busy--;
					break;
				}
			}
		}
	}

	void progress_options_serial(size_t &amount_to_do)
	{
		// Start and progress operations one at a time
		for(std::vector<QueueEntry *>::iterator entry = ongoing.begin(); entry != ongoing.end();
		    entry++)
		{
			(*entry)->start();
			while(!(*entry)->done())
			{
				(*entry)->progress();
			}
			amount_to_do--;
			busy--;
		}
		ongoing.clear();
	}
};

#endif