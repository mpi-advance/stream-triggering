#include "queues/CXIQueue.hpp"

#include "safety/gpu.hpp"
#include "safety/libfabric.hpp"

CompletionBufferFactory CXIQueue::my_buffer;
DeferredWorkQueue       CXIQueue::my_queue;

std::vector<size_t> populate_completion_vector()
{
    std::vector<size_t> new_vec(CompletionBufferFactory::MAX_COMP_VALUES);
    std::iota(new_vec.begin(), new_vec.end(), 1);
    return new_vec;
}

std::vector<size_t> CompletionBufferFactory::completion_addrs =
    populate_completion_vector();

void LibfabricInstance::initialize_libfabric()
{
    // Hints will be freed by helper call below
    struct fi_info* hints;
    hints = fi_allocinfo();

    // set our requirements for the providers
    hints->caps = FI_SHARED_AV | FI_RMA | FI_REMOTE_WRITE | FI_MSG | FI_WRITE | FI_HMEM |
                  FI_TRIGGER;
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

    /* Code specific to tioga -- ADD REASON */
#ifdef USE_GFX90A
    int device = -1;
    force_gpu(hipGetDevice(&device));
    int pci_bus_id = -1;
    force_gpu(hipDeviceGetAttribute(&pci_bus_id, hipDeviceAttributePciBusId, device));

    while (fi != nullptr)
    {
        Print::out(fi->nic->bus_attr->attr.pci.domain_id,
                   std::to_string(fi->nic->bus_attr->attr.pci.bus_id),
                   std::to_string(fi->nic->bus_attr->attr.pci.device_id));

        if ((int)fi->nic->bus_attr->attr.pci.bus_id == (pci_bus_id - 1) ||
            (int)fi->nic->bus_attr->attr.pci.bus_id == (pci_bus_id + 4))
        {
            Print::out("FOUND!");
            break;
        }
        fi = fi->next;
    }
#endif

    if (fi == nullptr)
    {
        throw std::runtime_error("Unable to select FI provider");
    }

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
    av_attr.count             = comm_size;
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

    recv_ctr = alloc_counter();
    check_libfabric(fi_ep_bind(ep, &(recv_ctr)->fid, FI_RECV));
}

void LibfabricInstance::initialize_peer_addresses(MPI_Comm comm_base)
{
    Print::out("Doing an allgather to get", comm_size, "ranks.");
    // Get our "cxi address"
    static constexpr size_t name_array_len = 100;
    char                    name[name_array_len];
    size_t                  _array_size = name_array_len;
    check_libfabric(fi_getname(&(ep)->fid, name, &_array_size));

    // All other ranks
    char* all_names = new char[_array_size * comm_size];
    memset(all_names, 0, _array_size * comm_size * sizeof(char));
    force_mpi(MPI_Allgather(name, 4, MPI_CHAR, all_names, 4, MPI_CHAR, comm_base));

    peers.resize(comm_size, 0);
    check_libfabric(fi_av_insert(av, all_names, comm_size, peers.data(), 0, 0));

    delete[] all_names;
}

void CXIQueue::prepare_cxi_mr_key(Request& req)
{
    /* New request that prepares the user's request to be enqueued later*/
    std::unique_ptr<CXIRequest> converted_request = nullptr;

    /* Data to be exchanged (receiver sends this, sender receives this)
     * 1. MR key for receive buffer
     * 2. Remote completion buffer mr key
     * 3. Remote completion buffer offset
     * TODO: UPDATE FOR NEW LISTS OF 5 ITEMS
     */
    if (Communication::Operation::RECV == req.operation)
    {
        converted_request = std::make_unique<CXIRecvOneSided>(req, my_buffer, libfab);
    }
    else if (Communication::Operation::RSEND >= req.operation)
    {
        converted_request = std::make_unique<CXISend>(req, my_buffer, libfab, my_queue,
                                                      libfab.get_peer(my_rank));
    }
    else
    {
        throw std::runtime_error("Operation not supported");
    }

    request_map.insert(std::make_pair(req.getID(), std::move(converted_request)));
}

LibfabricInstance::~LibfabricInstance()
{
    check_libfabric(fi_close(&(ep)->fid));
    check_libfabric(fi_close(&(recv_ctr)->fid));
    check_libfabric(fi_close(&(av)->fid));
    check_libfabric(fi_close(&(rxcq)->fid));
    check_libfabric(fi_close(&(txcq)->fid));
    check_libfabric(fi_close(&(domain)->fid));
    check_libfabric(fi_close(&(fabric)->fid));
}

__global__ void check_buffers(size_t* array, size_t array_len, int rank)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= array_len)
        return;

    printf("<GPU %d> Completion index %zu: %zu\n", rank, index, array[index]);
}

__global__ void sleep_kernel()
{
    auto a = clock64() + 1'000'000'000;
    while(clock64() < a)
    {
        // Party;
    }
}

__global__ void add_to_counter(uint64_t* cntr, size_t value)
{
    *cntr = value;
}

__global__ void wait_on_completion(volatile size_t* comp_addr, size_t goal_value)
{
    size_t curr_value = *comp_addr;
    while (curr_value < goal_value)
    {
        curr_value = *comp_addr;
    }
    //printf("Init: %zu Goal: %zu\n", init_value, goal_value);
}

__global__ void flush_buffer()
{
    asm volatile("buffer_wbl2");
}

void CXIRequest::wait_gpu(hipStream_t* the_stream)
{
    Print::out("<E> This request will wait for:", num_times_started, "at",
               completion_buffer.address);
    wait_on_completion<<<1, 1, 0, *the_stream>>>((size_t*)completion_buffer.address,
                                                 num_times_started);
}

void CXISend::start_gpu(hipStream_t* the_stream, Threshold& threshold,
                        CXICounter& trigger_cntr)
{
    if (Operation::RSEND != base_req.operation)
    {
        Print::out("<E> Starting kernel to wait on CTS;",
                   (size_t*)protocol_buffer.address, num_times_started);
        wait_on_completion<<<1, 1, 0, *the_stream>>>((size_t*)protocol_buffer.address,
                                                     num_times_started);
        //sleep_kernel<<<1, 1, 0, *the_stream>>>();
    }

    if (threshold.value() == threshold.counter_value())
    {
        Print::out("Skipping kernel for trigger -- counter already there");
        return;
    }

    size_t    counter_bump = threshold.equalize_counter();
    uint64_t* cntr_addr    = (uint64_t*)trigger_cntr.gpu_mmio_addr;
    Print::out("<E> Launching a kernel to bump a counter by", counter_bump);
    add_to_counter<<<1, 1, 0, *the_stream>>>(cntr_addr, counter_bump);
}

void CXIRecvOneSided::start_gpu(hipStream_t* the_stream, Threshold& threshold,
                                CXICounter& trigger_cntr)
{
    if (Operation::RSEND != protocol_buffer.operation)
    {
        if (threshold.value() == threshold.counter_value())
        {
            Print::out("Skipping kernel for CTS trigger -- counter already there");
            return;
        }

        size_t    counter_bump = threshold.equalize_counter();
        uint64_t* cntr_addr    = (uint64_t*)trigger_cntr.gpu_mmio_addr;
        Print::out("<E> Enqueueing triggering kernel for CTS! Bump:", counter_bump);

        add_to_counter<<<1, 1, 0, *the_stream>>>(cntr_addr, counter_bump);
    }
}

void CXIQueue::enqueue_waitall()
{
    // check_buffers<<<1, 4, 0, *the_stream>>>((size_t*)my_buffer.buffer, 4, my_rank);
    for (auto req : active_requests)
    {
        Print::out("Waiting on request:", req);
        request_map.at(req)->wait_gpu(the_stream);
    }
    active_requests.clear();
    // check_buffers<<<1, 4, 0, *the_stream>>>((size_t*)my_buffer.buffer, 4, my_rank);
    // sleep_kernel<<<1, 1, 0, *the_stream>>>();
}

void CXIQueue::flush_memory(hipStream_t* the_stream)
{
    Print::out("<E> Enqueuing buffer_wbl2 kernel");
    flush_buffer<<<1, 1, 0, *the_stream>>>();
}
