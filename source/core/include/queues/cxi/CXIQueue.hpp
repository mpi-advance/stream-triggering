#ifndef ST_CXI_QUEUE
#define ST_CXI_QUEUE

#include <map>
#include <numeric>
#include <vector>

#include "queues/cxi/libfabric_wrappers.hpp"
#include "queues/cxi/requests/request_base.hpp"
#include "misc/print.hpp"
#include "queues/HIPQueue.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

class CXIQueue : public HIPQueue
{
public:
    using CXIObjects = std::unique_ptr<CXIRequest>;

    CXIQueue(hipStream_t* stream_addr) : HIPQueue(stream_addr), comm_base(MPIS_COMM_WORLD)
    {
        Print::out("CXI Queue init-ed");
        force_mpi(MPI_Comm_rank(comm_base, &my_rank));
        Print::out("Starting MPI Comm Dupes");
        force_mpi(MPI_Comm_dup(comm_base, &protocol_phase));
        force_mpi(MPI_Comm_dup(comm_base, &match_phase_a));
        force_mpi(MPI_Comm_dup(comm_base, &match_phase_b));
        Print::out("Starting Allreduce to get CXI address data");
        libfab.initialize(match_phase_a);
        Print::out("Creating on-node communicator");
        force_mpi(MPI_Comm_split_type(comm_base, MPI_COMM_TYPE_SHARED, my_rank,
                                      MPI_INFO_NULL, &on_node_peers));

        // Global counter for when requests can share a triggering counter
        the_gpu_counter = std::make_unique<CXICounter>(libfab);

        // Register MR
        my_buffer.register_mr(libfab);
    }

    ~CXIQueue()
    {
        MPI_Barrier(comm_base);
        MPI_Comm_free(&protocol_phase);
        MPI_Comm_free(&match_phase_a);
        MPI_Comm_free(&match_phase_b);
        MPI_Comm_free(&on_node_peers);
        the_gpu_counter.reset();
        request_map.clear();
        my_buffer.free_mr();
    }

    void enqueue_operation(std::shared_ptr<Request> request) override
    {
        enqueue_startall({request});
    }

    void enqueue_startall(std::vector<std::shared_ptr<Request>> requests) override;

    void enqueue_waitall() override;

    void host_wait() override
    {
        Queue::host_wait();
        Print::out("Waiting on device!");
        force_gpu(hipStreamSynchronize(*my_stream));
    }

    void initiate_match(std::vector<std::shared_ptr<Request>> requests) override
    {
        for (auto& req : requests)
        {
            if (Communication::Operation::BARRIER <= req->operation)
            {
                HIPQueue::initiate_match({req});
            }
            else
            {
                exchange_protocol(*req);
            }
        }
    }

    void finalize_match(std::vector<std::shared_ptr<Request>> requests) override
    {
        for (auto& req : requests)
        {
            if (Communication::Operation::BARRIER > req->operation)
            {
                prepare_cxi_mr_key(*req);
            }
        }
        Print::out("CXI Queue done with matching on its end");
        Queue::finalize_match(requests);
    }

private:
    void exchange_protocol(Request&);
    void prepare_cxi_mr_key(Request&);
    void flush_memory();

    void inline start_request(CXIObjects& cxi_stuff) {}

    // Persistent Libfabric objects
    LibfabricInstance libfab;

    // Peer information
    MPI_Comm comm_base;
    int      my_rank;
    MPI_Comm protocol_phase;
    MPI_Comm match_phase_a;
    MPI_Comm match_phase_b;
    MPI_Comm on_node_peers;

    // Matching stuff
    std::map<size_t, std::unique_ptr<int>> active_match_requests;

    // Map of Request ID to CXIObject (counters, mr)
    std::map<size_t, CXIObjects> request_map;
    std::vector<size_t>          active_requests;

    // Completion buffers
    CompletionBufferFactory my_buffer;

    // GPU Triggerable Counter
    std::unique_ptr<CXICounter> the_gpu_counter;
};

#endif
