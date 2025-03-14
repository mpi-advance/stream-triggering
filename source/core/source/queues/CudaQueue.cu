#include "misc/print.hpp"
#include "queues/CudaQueue.hpp"
#include "safety/cuda.hpp"
#include "safety/mpi.hpp"

CudaQueueEntry::CudaQueueEntry(std::shared_ptr<Request> req) : my_request(req)
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
        default:
            throw std::runtime_error("Invalid Request");
            break;
    }

    force_cuda(cuMemHostAlloc(
        (void**)&start_location, sizeof(int64_t),
        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED));
    *start_location = 0;
    force_cuda(cuMemHostGetDevicePointer(&start_dev, start_location, 0));
    force_cuda(cudaHostAlloc(
        (void**)&wait_location, sizeof(int64_t),
        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED));
    *wait_location = 0;
    force_cuda(cuMemHostGetDevicePointer(&wait_dev, wait_location, 0));
}

CudaQueueEntry::~CudaQueueEntry()
{
    std::cout << "Entry going away!" << std::endl;
    check_cuda(cudaFreeHost(start_location));
    check_cuda(cudaFreeHost(wait_location));
}

void CudaQueueEntry::prepare()
{
    // Do nothing?
}

void CudaQueueEntry::start()
{
    Print::out("Waiting for GPU to tell us to start!");
    while ((*start_location) != 1)
    {
        std::this_thread::yield();
    }
    check_mpi(MPI_Start(&mpi_request));
    Print::out("GPU says we should start: ", *start_location);
}

bool CudaQueueEntry::done()
{
    int value = 0;
    check_mpi(MPI_Test(&mpi_request, &value, MPI_STATUS_IGNORE));
    if (value)
    {
        (*wait_location) = 1;
        Print::out("Waiting value set!");
    }
    return value;
}

void CudaQueueEntry::launch_start_kernel(CUstream the_stream)
{
    Print::out("Queueing start kernel!");
    force_cuda(cuStreamWriteValue64(the_stream, start_dev, 1, 0));
}

void CudaQueueEntry::launch_wait_kernel(CUstream the_stream)
{
    Print::out("Queueing wait kernel!");
    force_cuda(cuStreamWaitValue64(the_stream, wait_dev, 1, 0));
}

CudaQueue::CudaQueue(cudaStream_t* stream)
    : thr(&CudaQueue::progress, this), my_stream(stream)
{
    // force_cuda(cuInit(0));
    // force_cuda(cudaSetDevice(0));
}

CudaQueue::~CudaQueue()
{
    shutdown = true;
    thr.join();
}

void CudaQueue::progress()
{
    while (!shutdown)
    {
        while (start_cntr.load() > 0 || wait_cntr.load() > 0)
        {
            {
                std::scoped_lock<std::mutex> incoming_lock(queue_guard);
                for (CudaQueueEntry* entry : s_ongoing)
                {
                    entry->start();
                    start_cntr--;
                    w_ongoing.push(entry);
                }
                s_ongoing.clear();
            }

            for (size_t i = 0; i < w_ongoing.size(); ++i)
            {
                CudaQueueEntry* entry = w_ongoing.front();
                if (entry->done())
                {
                    wait_cntr--;
                    w_ongoing.pop();
                }
                else
                {
                    break;
                }
            }

            if (shutdown)
                break;
        }

        std::this_thread::yield();
    }
}

void CudaQueue::enqueue_operation(std::shared_ptr<Request> qe)
{
    if (wait_cntr.load() > 0)
        Print::out("WARNING!");

    CudaQueueEntry* cqe = new CudaQueueEntry(qe);
    cqe->launch_start_kernel(*my_stream);
    start_cntr++;
    entries.push_back(cqe);

    std::scoped_lock<std::mutex> incoming_lock(queue_guard);
    s_ongoing.push_back(cqe);
}

void CudaQueue::enqueue_waitall()
{
    Print::out("enqueue waiting");
    while (start_cntr.load())
    {
        // Do nothing
    }
    Print::out("Done enqueue waiting");

    for (CudaQueueEntry* entry : entries)
    {
        entry->launch_wait_kernel(*my_stream);
        wait_cntr++;
        Print::out("Waitng for 1 entry");
        while (wait_cntr.load())
        {
            // do nothing
        }
    }
    entries.clear();
}

void CudaQueue::host_wait()
{
    while (start_cntr.load() || wait_cntr.load())
    {
        // Do nothing.
    }
}