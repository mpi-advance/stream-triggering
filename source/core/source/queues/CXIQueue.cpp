#include "queues/CXIQueue.hpp"

#include "abstract/match.hpp"
#include "safety/hip.hpp"
#include "safety/libfabric.hpp"

CompletionBuffer  CXIQueue::my_buffer;
DeferredWorkQueue CXIQueue::my_queue;

void CXIQueue::libfabric_setup(int num_ranks)
{
    // Hints will be freed by helper call below
    struct fi_info* hints;
    hints = fi_allocinfo();

    // set our requirements for the providers
    hints->caps = FI_SHARED_AV | FI_RMA | FI_REMOTE_WRITE | FI_MSG | FI_WRITE |
                  FI_HMEM | FI_TRIGGER;
    hints->addr_format   = FI_ADDR_CXI;
    hints->ep_attr->type = FI_EP_RDM;  // Must use for connection-less
    hints->tx_attr->caps =
        FI_SHARED_AV | FI_RMA | FI_WRITE | FI_MSG | FI_HMEM | FI_TRIGGER;
    hints->rx_attr->caps =
        FI_SHARED_AV | FI_RMA | FI_REMOTE_WRITE | FI_MSG | FI_HMEM | FI_TRIGGER;
    hints->domain_attr->mr_mode =
        FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_LOCAL | FI_MR_HMEM;
    hints->mode = FI_CONTEXT;

    // Get information on available providers given our requirements in
    // hints
    force_libfabric(fi_getinfo(FI_VERSION(1, 15), 0, 0, 0, hints, &fi));
    fi_freeinfo(hints);  // deallocate hints

    // Create the fabric from the provider information
    force_libfabric(fi_fabric(fi->fabric_attr, &fabric, 0));

    // Make Domain
    force_libfabric(fi_domain(fabric, fi, &domain, 0));

    // Make some completion queues!
    struct fi_cq_attr cq_attr = {};
    cq_attr.size              = fi->tx_attr->size;
    cq_attr.flags             = 0;
    cq_attr.format            = FI_CQ_FORMAT_DATA;
    cq_attr.wait_obj          = FI_WAIT_UNSPEC;
    cq_attr.signaling_vector  = 0;
    cq_attr.wait_cond         = FI_CQ_COND_NONE;
    cq_attr.wait_set          = 0;
    force_libfabric(fi_cq_open(domain, &cq_attr, &txcq, 0));

    cq_attr.size = fi->rx_attr->size;
    force_libfabric(fi_cq_open(domain, &cq_attr, &rxcq, 0));

    // Make Address Vector
    struct fi_av_attr av_attr = {};
    av_attr.type              = FI_AV_TABLE;
    av_attr.count             = num_ranks;
    av_attr.flags             = FI_SYMMETRIC;
    force_libfabric(fi_av_open(domain, &av_attr, &av, 0));

    // Endpoint setup
    force_libfabric(fi_endpoint(domain, fi, &ep, 0));

    // Bind T/R Completion Queues to EP -- Note flags!
    check_libfabric(fi_ep_bind(ep, &(txcq)->fid, FI_TRANSMIT));
    check_libfabric(fi_ep_bind(ep, &(rxcq)->fid, FI_RECV));
    // Bind EP to AV before enabling
    check_libfabric(fi_ep_bind(ep, &(av)->fid, 0));
    check_libfabric(fi_enable(ep));

    recv_ctr = CXICounter::alloc_counter(domain);
    check_libfabric(fi_ep_bind(ep, &(recv_ctr)->fid, FI_RECV));

    // Register MR
    my_buffer.register_mr(domain, ep);

    // Register progress counter
    my_queue.regsiter_counter(recv_ctr);
}

void CXIQueue::peer_setup(int size)
{
    // Get our "cxi address"
    static constexpr size_t name_array_len = 100;
    char                    name[name_array_len];
    size_t                  _array_size = name_array_len;
    check_libfabric(fi_getname(&(ep)->fid, name, &_array_size));

    // All other ranks
    char* all_names = new char[_array_size * size];
    memset(all_names, 0, _array_size * size * sizeof(char));
    force_mpi(
        MPI_Allgather(name, 4, MPI_CHAR, all_names, 4, MPI_CHAR, comm_base));
    check_libfabric(fi_av_insert(av, all_names, size, peers.data(), 0, 0));

    delete[] all_names;
}

void CXIQueue::prepare_cxi_mr_key(Request& req)
{
    MPI_Datatype the_type = req.datatype;
    int          size     = -1;
    check_mpi(MPI_Type_size(the_type, &size));
    int total_size = size * req.count;

    size_t req_id = req.getID();

    Buffer local_completion = my_buffer.alloc_buffer();
    /* The data_area buffer does not have an MR key yet,
     * and will always have an offset of 0 since it's the user's
     * buffer from MPI! Note that only the receiver will ever
     * introduce an MR for this -- the sender doesn't have
     * an MR key for the data it is sending */
    Buffer data_area = Buffer(req.buffer, total_size, 0, 0);

    /* Data exchanged (receiver sends this, sender receives this)
     * 1. Data buffer mr key
     * 2. Remote completion buffer mr
     * 3. Remote completion buffer offset  */
    if (Communication::Operation::RECV == req.operation)
    {
        /*  Will add MR key to data_buffer! */
        std::unique_ptr<CXIRecvOneSided> temp_object =
            std::make_unique<CXIRecvOneSided>(local_completion, data_area,
                                              domain, ep);

        std::vector<uint64_t> matching_data(3);
        matching_data.at(0) = data_area.mr_key;
        matching_data.at(1) = local_completion.mr_key;
        matching_data.at(2) = local_completion.offset;
        Communication::OneSideMatch::give(matching_data, req.peer, req.comm);
        req.toggle_match();

        request_map.insert(std::make_pair(req_id, std::move(temp_object)));
    }
    else if (Communication::Operation::SEND == req.operation)
    {
        std::vector<uint64_t> peer_mr_data =
            Communication::OneSideMatch::take<uint64_t>(3, req.peer, req.comm);
        req.toggle_match();

        data_area.mr_key = peer_mr_data.at(0);
        Buffer remote_completion =
            Buffer(0, CompletionBuffer::DEFAULT_SIZE, peer_mr_data.at(1),
                   peer_mr_data.at(2));

        constexpr int string_size = 10;
        char          info_key[]  = "MPIS_GPU_MEM_TYPE";
        char          value[string_size];
        int           flag = 0;
        // Pre MPI-4.0
        force_mpi(MPI_Info_get(req.info, info_key, string_size, value, &flag));

        if (0 == strcmp(value, "COARSE"))
        {
            using SendType =
                CXISend<CommunicationType::ONE_SIDED, GPUMemoryType::COARSE>;
            std::unique_ptr<SendType> temp_object = std::make_unique<SendType>(
                local_completion, remote_completion, data_area, domain, ep,
                my_queue, peers.at(req.peer), peers.at(my_rank));

            request_map.insert(std::make_pair(req_id, std::move(temp_object)));
        }
        else
        {
            using SendType =
                CXISend<CommunicationType::ONE_SIDED, GPUMemoryType::FINE>;
            std::unique_ptr<SendType> temp_object = std::make_unique<SendType>(
                local_completion, remote_completion, data_area, domain, ep,
                my_queue, peers.at(req.peer), peers.at(my_rank));

            request_map.insert(std::make_pair(req_id, std::move(temp_object)));
        }
    }
    else
    {
        throw std::runtime_error("Operation not supported");
    }
}

void CXIQueue::libfabric_teardown()
{
    the_gpu_counter.reset();
    request_map.clear();
    my_buffer.free_mr();
    check_libfabric(fi_close(&(ep)->fid));
    check_libfabric(fi_close(&(recv_ctr)->fid));
    check_libfabric(fi_close(&(av)->fid));
    check_libfabric(fi_close(&(rxcq)->fid));
    check_libfabric(fi_close(&(txcq)->fid));
    check_libfabric(fi_close(&(domain)->fid));
    check_libfabric(fi_close(&(fabric)->fid));
}

__global__ void add_to_counter(uint64_t* cntr, size_t value)
{
    *cntr = value;
}

__global__ void wait_on_completion(volatile size_t* comp_addr,
                                   size_t           goal_value)
{
    size_t curr_value = *comp_addr;
    while (curr_value < goal_value)
    {
        curr_value = *comp_addr;
    }
}

__global__ void flush_buffer()
{
    asm volatile("buffer_wbl2");
}

void CXIWait::wait_gpu(hipStream_t* the_stream)
{
    wait_on_completion<<<1, 1, 0, *the_stream>>>(
        (size_t*)completion_buffer.address, num_times_started);
}

template <CommunicationType MODE, GPUMemoryType G>
void CXISend<MODE, G>::start_gpu(hipStream_t* the_stream, Threshold& threshold,
                                 CXICounter& trigger_cntr)
{
    if constexpr (GPUMemoryType::COARSE == G)
    {
        flush_buffer<<<1, 1, 0, *the_stream>>>();
    }

    if(threshold.value() == threshold.counter_value())
    {
        // No need to trigger counter, as eventually it should be what we want.
        return;
    }

    size_t counter_bump = threshold.equalize_counter();
    uint64_t* cntr_addr = (uint64_t*)trigger_cntr.gpu_mmio_addr;
    add_to_counter<<<1, 1, 0, *the_stream>>>(cntr_addr, counter_bump);
}

void CXIQueue::enqueue_waitall()
{
    for (auto req : active_requests)
    {
        request_map.at(req)->wait_gpu(the_stream);
    }
    active_requests.clear();
}
