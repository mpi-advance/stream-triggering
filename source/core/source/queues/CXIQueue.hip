#include "queues/CXIQueue.hpp"
#include "safety/hip.hpp"
#include "safety/libfabric.hpp"

void CXIQueue::libfabric_setup()
{
    // Hints will be freed by helper call below
    struct fi_info* hints;
    hints = fi_allocinfo();

    // set our requirements for the providers
    hints->caps = FI_SHARED_AV | FI_RMA | FI_REMOTE_READ | FI_REMOTE_WRITE |
                  FI_RMA_EVENT | FI_MSG | FI_WRITE | FI_HMEM | FI_TRIGGER;
    hints->addr_format   = FI_ADDR_CXI;
    hints->ep_attr->type = FI_EP_RDM;  // Must use for connection-less
    hints->tx_attr->caps = FI_SHARED_AV | FI_RMA | FI_WRITE | FI_RMA_EVENT |
                           FI_MSG | FI_HMEM | FI_TRIGGER;
    hints->rx_attr->caps = FI_SHARED_AV | FI_RMA | FI_WRITE | FI_REMOTE_WRITE |
                           FI_RMA_EVENT | FI_MSG | FI_HMEM | FI_TRIGGER;
    hints->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED |
                                  FI_MR_RMA_EVENT | FI_MR_LOCAL | FI_MR_HMEM;
    hints->mode = FI_CONTEXT;

    // Get information on available providers given our requirements in
    // hints
    force_libfabric(fi_getinfo(FI_VERSION(1, 15), 0, 0, 0, hints, &fi));
    fi_freeinfo(hints);  // deallocate hints

    // Create the fabric from the provider information
    check_libfabric(fi_fabric(fi->fabric_attr, &fabric, 0));

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
    av_attr.count             = 2;
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

    // Get our "cxi address"
    size_t _array_size = name_array_len;
    check_libfabric(fi_getname(&(ep)->fid, name, &_array_size));
}

void CXIQueue::peer_setup(int rank)
{
    // Do a setup
    char othername[name_array_len];
    memset(othername, 0, name_array_len * sizeof(char));
    size_t other_len = name_array_len;

    int my_rank;
    force_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    if (my_rank > rank)
    {
        force_mpi(MPI_Send(name, name_array_len, MPI_CHAR, rank, OOB_TAG,
                           MPI_COMM_WORLD));
        force_mpi(MPI_Recv(othername, name_array_len, MPI_CHAR, rank, OOB_TAG,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    else
    {
        force_mpi(MPI_Recv(othername, name_array_len, MPI_CHAR, rank, OOB_TAG,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        force_mpi(MPI_Send(name, name_array_len, MPI_CHAR, rank, OOB_TAG,
                           MPI_COMM_WORLD));
    }

    // Try to connect to peer?
    fi_addr_t* partner = &peers[rank];
    check_libfabric(fi_av_insert(av, othername, 1, partner, 0, 0));
}

void CXIQueue::prepare_cxi_mr_key(Request& req)
{
    MPI_Datatype the_type = req.datatype;
    int          size     = -1;
    check_mpi(MPI_Type_size(the_type, &size));
    int total_size = size * req.count;

    size_t req_id = req.getID();
    if (Communication::Operation::RECV == req.operation)
    {
        std::unique_ptr<CXIRecvOneSided> temp_object =
            std::make_unique<CXIRecvOneSided>(domain, ep, req.buffer,
                                              total_size);
        req.match(temp_object->get_mr_key());
        request_map.insert(std::make_pair(req_id, std::move(temp_object)));
    }
    else if (Communication::Operation::SEND == req.operation)
    {
        req.match<size_t>(0);
        const std::optional<MatchData>& match_info = req.getMatch();
        if (std::nullopt == match_info)
            throw std::runtime_error("Request was not matched properly!");
        uint64_t match_value =
            *((uint64_t*)(match_info->get_peer_match_data()));

        constexpr int string_size = 10;
        char          info_key[]  = "MPIS_GPU_MEM_TYPE";
        char          value[string_size];
        int           string_res = string_size;
        int           flag       = 0;
        // Pre MPI-4.0
        force_mpi(MPI_Info_get(req.info, info_key, string_size, value, &flag));

        if (0 == strcmp(value, "COARSE"))
        {
            using SendType =
                CXISend<CommunicationType::ONE_SIDED, GPUMemoryType::COARSE>;
            std::unique_ptr<SendType> temp_object =
                std::make_unique<SendType>(domain, ep);
            temp_object->fill_message(peers.at(req.peer), req.buffer,
                                      total_size, match_value);
            request_map.insert(std::make_pair(req_id, std::move(temp_object)));
        }
        else
        {
            using SendType =
                CXISend<CommunicationType::ONE_SIDED, GPUMemoryType::FINE>;
            std::unique_ptr<SendType> temp_object =
                std::make_unique<SendType>(domain, ep);
            temp_object->fill_message(peers.at(req.peer), req.buffer,
                                      total_size, match_value);
            request_map.insert(std::make_pair(req_id, std::move(temp_object)));
        }
    }
}

void CXIQueue::libfabric_teardown() {}

__global__ void add_to_counter(uint64_t* cntr)
{
    *cntr = 1;
}

__global__ void wait_on_counter(volatile uint64_t* cntr_addr, uint64_t value)
{
    uint64_t cntr_value = ((*cntr_addr) & ((1ULL << 48) - 1));
    while (cntr_value != value)
    {
        cntr_value = ((*cntr_addr) & ((1ULL << 48) - 1));
    }
}

__global__ void flush_buffer()
{
    asm volatile("buffer_wbl2");
}

void CXIWait::wait_gpu(hipStream_t* the_stream)
{
    uint64_t* cntr_addr = (uint64_t*)completion_counter.gpu_wb_buffer;
    wait_on_counter<<<1, 1, 0, *the_stream>>>(cntr_addr, threshold);
}

template <GPUMemoryType G>
void CXITrigger<G>::start_gpu(hipStream_t* the_stream)
{
    if constexpr (GPUMemoryType::COARSE == G)
    {
        flush_buffer<<<1, 1, 0, *the_stream>>>();
    }
    uint64_t* cntr_addr = (uint64_t*)trigger_counter.gpu_mmio_addr;
    add_to_counter<<<1, 1, 0, *the_stream>>>(cntr_addr);
}

void CXIWait::wait_host()
{
    size_t base  = threshold;
    size_t value = fi_cntr_read(completion_counter.counter);
    while (value != base)
    {
        value = fi_cntr_read(completion_counter.counter);
    }
}

void CXIQueue::enqueue_waitall()
{
    for (auto req : active_requests)
    {
        request_map.at(req)->wait_gpu(the_stream);
        request_map.at(req)->wait_host();
    }
    active_requests.clear();
}
