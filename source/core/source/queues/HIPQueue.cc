#include "queues/HIPQueue.hpp"
#include "safety/hip.hpp"
#include "safety/mpi.hpp"

namespace Print
{
static int rank = -1;

void initialize_rank()
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

template <typename T, typename... Args>
void print_out_r(const T& arg, Args&&... args)
{
    std::cout << arg << " ";
    if constexpr (sizeof...(Args))  // If still have other parameters
        print_out_r(std::forward<Args>(args)...);
    else
        std::cout << std::endl;
}

template <bool UseRanks = true, typename... Args>
void out(Args&&... args)
{
#ifndef NDEBUG
    if constexpr (UseRanks)
    {
        std::cout << "Rank: " << Print::rank << " - ";
    }
    print_out_r(std::forward<Args>(args)...);
#endif
}
}  // namespace Print

HIPQueueEntry::HIPQueueEntry(std::shared_ptr<Request> req) : my_request(req)
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

    force_hip(hipHostMalloc(
        (void**)&start_location, sizeof(int64_t), 0));
    *start_location = 0;
    force_hip(hipHostGetDevicePointer(&start_dev, start_location, 0));
    force_hip(hipHostMalloc(
        (void**)&wait_location, sizeof(int64_t), 0));
    *wait_location = 0;
    force_hip(hipHostGetDevicePointer(&wait_dev, wait_location, 0));
}

HIPQueueEntry::~HIPQueueEntry()
{
    std::cout << "Entry going away!" << std::endl;
    check_hip(hipHostFree(start_location));
    check_hip(hipHostFree(wait_location));
}

void HIPQueueEntry::prepare()
{
    // Do nothing?
}

void HIPQueueEntry::start()
{
    Print::out("Waiting for GPU to tell us to start!");
    while ((*start_location) != 1)
    {
        std::this_thread::yield();
    }
    check_mpi(MPI_Start(&mpi_request));
    Print::out("GPU says we should start: ", *start_location);
}

bool HIPQueueEntry::done()
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

void HIPQueueEntry::launch_start_kernel(hipStream_t the_stream)
{
    Print::out("Queueing start kernel!");
    force_hip(hipStreamWriteValue64(the_stream, start_dev, 1, 0));
}

void HIPQueueEntry::launch_wait_kernel(hipStream_t the_stream)
{
    Print::out("Queueing wait kernel!");
    force_hip(hipStreamWaitValue64(the_stream, wait_dev, 1, 0));
}

HIPQueue::HIPQueue(hipStream_t* stream)
    : thr(&HIPQueue::progress, this), my_stream(stream)
{
    // force_hip(cuInit(0));
    // force_hip(hipSetDevice(0));
    Print::initialize_rank();
}

HIPQueue::~HIPQueue()
{
    shutdown = true;
    thr.join();
}

void HIPQueue::progress()
{
    while (!shutdown)
    {
        while (start_cntr.load() > 0 || wait_cntr.load() > 0)
        {
            {
                std::scoped_lock<std::mutex> incoming_lock(queue_guard);
                for (HIPQueueEntry* entry : s_ongoing)
                {
                    entry->start();
                    start_cntr--;
                    w_ongoing.push(entry);
                }
                s_ongoing.clear();
            }

            for (size_t i = 0; i < w_ongoing.size(); ++i)
            {
                HIPQueueEntry* entry = w_ongoing.front();
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

void HIPQueue::enqueue_operation(std::shared_ptr<Request> qe)
{
    if (wait_cntr.load() > 0)
        Print::out("WARNING!");

    HIPQueueEntry* cqe = new HIPQueueEntry(qe);
    cqe->launch_start_kernel(*my_stream);
    start_cntr++;
    entries.push_back(cqe);

    std::scoped_lock<std::mutex> incoming_lock(queue_guard);
    s_ongoing.push_back(cqe);
}

void HIPQueue::enqueue_waitall()
{
    Print::out("enqueue waiting");
    while (start_cntr.load())
    {
        // Do nothing
    }
    Print::out("Done enqueue waiting");

    for (HIPQueueEntry* entry : entries)
    {
        entry->launch_wait_kernel(*my_stream);
        wait_cntr++;
        Print::out("Waitng for 1 entry");
        while(wait_cntr.load())
        {
            // do nothing
        }
    }
    entries.clear();
}

void HIPQueue::host_wait()
{
    while (start_cntr.load() || wait_cntr.load())
    {
        // Do nothing.
    }
}

void HIPQueue::match(std::shared_ptr<Request> request)
{
    // Normal matching
    request->match();

    auto& match_info = request->getMatch();
    if (std::nullopt == match_info)
        throw std::runtime_error("Request was not matched properly!");
}