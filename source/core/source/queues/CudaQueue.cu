#include "queues/CudaQueue.hpp"
#include "safety/cuda.hpp"
#include "safety/mpi.hpp"

CudaQueueEntry::CudaQueueEntry(MPI_Request req) : QueueEntry(req)
{
	force_cuda(cuMemHostAlloc((void **) &start_location, sizeof(int), CU_MEMHOSTALLOC_PORTABLE));
	force_cuda(cuMemHostGetDevicePointer(&start_dev, start_location, 0));
	force_cuda(cudaHostAlloc((void **) &wait_location, sizeof(int), CU_MEMHOSTALLOC_PORTABLE));
	force_cuda(cuMemHostGetDevicePointer(&wait_dev, wait_location, 0));
}

CudaQueueEntry::~CudaQueueEntry()
{
	check_cuda(cudaFreeHost(start_location));
	check_cuda(cudaFreeHost(wait_location));
}

void CudaQueueEntry::prepare()
{
	// Do nothing?
}

void CudaQueueEntry::start()
{
	std::cout << "Starting" << std::endl;
	while((*start_location) != 1)
	{
		// Do nothing
	}
	check_mpi(MPI_Start(&my_request));
	std::cout << "Done: " << *start_location << std::endl;
}

bool CudaQueueEntry::done()
{
	int value = 0;
	check_mpi(MPI_Test(&my_request, &value, MPI_STATUS_IGNORE));
	if(value)
	{
		(*wait_location) = 1;
		std::cout << "Value set!" << std::endl;
	}
	return value;
}

void CudaQueueEntry::progress()
{
	done();
}

void CudaQueueEntry::launch_start_kernel(CUstream the_stream)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::cout << rank << " Queueing start!" << std::endl;
	force_cuda(cuStreamWriteValue64(the_stream, start_dev, 1, 0));
}

void CudaQueueEntry::launch_wait_kernel(CUstream the_stream)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::cout << rank << " Queueing wait!" << std::endl;
	force_cuda(cuStreamWaitValue64(the_stream, wait_dev, 1, 0));
}

CudaQueue::CudaQueue() : thr(&CudaQueue::progress, this)
{
	force_cuda(cuInit(0));
	force_cuda(cudaSetDevice(0));
	force_cuda(cudaStreamCreateWithFlags(&my_stream, cudaStreamNonBlocking));
}

CudaQueue::~CudaQueue()
{
	shutdown = true;
	thr.join();
	check_cuda(cudaStreamDestroy(my_stream));
}

void CudaQueue::progress()
{
	while(!shutdown)
	{
		while(busy.load() > 0)
		{
			{
				std::scoped_lock<std::mutex> incoming_lock(queue_guard);
				for(CudaQueueEntry *entry : s_ongoing)
				{
					entry->start();
					w_ongoing.push(entry);
				}
				s_ongoing.clear();
			}

			for(size_t i = 0; i < w_ongoing.size(); ++i)
			{
				CudaQueueEntry *entry = w_ongoing.front();
				if(entry->done())
				{
					busy--;
					w_ongoing.pop();
				}
				else
				{
					break;
				}
			}

			if(shutdown)
				break;
		}

		std::this_thread::yield();
	}
}

QueueEntry *CudaQueue::create_entry(MPI_Request req)
{
	return new CudaQueueEntry(req);
}

void CudaQueue::enqueue_operation(QueueEntry *qe)
{
	CudaQueueEntry *cqe = static_cast<CudaQueueEntry *>(qe);
	cqe->launch_start_kernel(my_stream);
	entries.push_back(qe);

	std::scoped_lock<std::mutex> incoming_lock(queue_guard);
	s_ongoing.push_back(cqe);
	busy++;
}

void CudaQueue::enqueue_waitall()
{
	for(QueueEntry *entry : entries)
	{
		CudaQueueEntry *cqe = static_cast<CudaQueueEntry *>(entry);
		cqe->launch_wait_kernel(my_stream);
	}
	entries.clear();
}

void CudaQueue::host_wait()
{
	while(busy.load())
	{
		// Do nothing.
	}
}