#include "queues/ThreadQueue.h"

#include "safety/mpi.hpp"

void ThreadQueueEntry::prepare()
{
	// Do nothing?
}

void ThreadQueueEntry::start()
{
	check_mpi(MPI_Start(&my_request));
}

bool ThreadQueueEntry::done()
{
	int value = 0;
	check_mpi(MPI_Test(&my_request, &value, MPI_STATUS_IGNORE));
	return value;
}

void ThreadQueueEntry::progress()
{
	done();
}

ThreadQueue::ThreadQueue() : thr(&ThreadQueue::progress, this) {}

ThreadQueue::~ThreadQueue()
{
	shutdown = true;
	thr.join();
}

void ThreadQueue::progress()
{
	size_t amount_to_do = 0;

	while(!shutdown)
	{
		if(amount_to_do == 0 && stop_counts.size() > 0)
		{
			{ // Scope of the lock
				std::scoped_lock<std::mutex> incoming_lock(queue_guard);
				amount_to_do = stop_counts.front();
				busy.store(amount_to_do);
				stop_counts.pop();
				ongoing.insert(ongoing.begin(), entries.begin(), entries.begin() + amount_to_do);
				entries.clear();
			}

			while(amount_to_do != 0 && !shutdown)
			{
				for(std::vector<QueueEntry *>::iterator entry_i = ongoing.begin();
				    entry_i != ongoing.end();
				    entry_i++)
				{
					QueueEntry *entry = *entry_i;
					entry->progress();
					if(entry->done())
					{
						ongoing.erase(entry_i);
						amount_to_do--;
						break;
					}
				}
			}

			busy.store(0);
		}
		else
		{
			std::this_thread::yield();
		}
	}
}

QueueEntry *ThreadQueue::create_entry(MPI_Request req)
{
	return new ThreadQueueEntry(req);
}

void ThreadQueue::enqueue_operation(QueueEntry *qe)
{
	qe->start();
	std::scoped_lock<std::mutex> incoming_lock(queue_guard);
	entries.push_back(qe);
}

void ThreadQueue::enqueue_waitall()
{
	std::scoped_lock<std::mutex> incoming_lock(queue_guard);
	stop_counts.push(entries.size());
	busy.store(1);
}

void ThreadQueue::host_wait()
{
	while(busy.load())
	{
		// Do nothing.
		// No lock because this function doesn't want to set
		// amount_to_do to any value, only read.
	}
}