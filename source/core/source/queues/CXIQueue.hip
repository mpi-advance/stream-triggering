#include "queues/CXIQueue.hpp"
#include "safety/gpu.hpp"
#include "safety/libfabric.hpp"

void LibfabricInstance::initialize_libfabric()
{
    // Hints will be freed by helper call below
    struct fi_info* hints;
    hints = fi_allocinfo();

    // set our requirements for the providers
    hints->caps = FI_RMA | FI_RMA_EVENT | FI_REMOTE_WRITE | FI_MSG | FI_WRITE | FI_HMEM |
                  FI_TRIGGER;
    hints->addr_format   = FI_ADDR_CXI;
    hints->ep_attr->type = FI_EP_RDM;  // Must use for connection-less
    hints->tx_attr->caps = FI_RMA | FI_WRITE | FI_MSG | FI_HMEM | FI_TRIGGER;
    hints->rx_attr->caps = FI_RMA | FI_REMOTE_WRITE | FI_MSG | FI_HMEM | FI_TRIGGER;
    hints->domain_attr->mr_mode =
        FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_LOCAL | FI_MR_HMEM;
    hints->mode = FI_CONTEXT;

    // Get information on available providers given our requirements in
    // hints
    force_libfabric(fi_getinfo(FI_VERSION(1, 15), 0, 0, 0, hints, &fi));

    select_fi_nic(fi);
    Print::out(fi->nic->bus_attr->attr.pci.domain_id,
               std::to_string(fi->nic->bus_attr->attr.pci.bus_id),
               std::to_string(fi->nic->bus_attr->attr.pci.device_id));

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

    progress_ctr = alloc_counter(false);
    check_libfabric(fi_ep_bind(ep, &(progress_ctr)->fid, FI_RECV));
}

void LibfabricInstance::select_fi_nic(fi_info*& info)
{
    /* Get device information */
    int device = -1;
    force_gpu(hipGetDevice(&device));
    int device_pci_bus_id    = -1;
    int device_pci_domain_id = -1;
    force_gpu(
        hipDeviceGetAttribute(&device_pci_bus_id, hipDeviceAttributePciBusId, device));
    force_gpu(hipDeviceGetAttribute(&device_pci_domain_id, hipDeviceAttributePciDomainID,
                                    device));

    auto validation_lambda = [&](fid_nic* nic_info) {
#if defined(USE_TIOGA) /* Code specific to tioga's NIC & GPU layout */
        return ((int)nic_info->bus_attr->attr.pci.bus_id == (device_pci_bus_id - 1) ||
                (int)nic_info->bus_attr->attr.pci.bus_id == (device_pci_bus_id + 4));

#elif defined(USE_TUOLUMNE)
        return ((int)nic_info->bus_attr->attr.pci.domain_id == device_pci_domain_id);
#else
        Print::always("Warning -- taking first CXI provider found!");
        return true;
#endif
    };

    while (info != nullptr)
    {
        Print::out(info->nic->bus_attr->attr.pci.domain_id,
                   std::to_string(info->nic->bus_attr->attr.pci.bus_id),
                   std::to_string(info->nic->bus_attr->attr.pci.device_id));

        if (validation_lambda(info->nic))
        {
            Print::out("Found acceptable nic!");
            break;
        }
        info = info->next;
    }

    if (info == nullptr)
    {
        throw std::runtime_error("Unable to select FI provider");
    }
}

void LibfabricInstance::initialize_peer_addresses(MPI_Comm comm)
{
    Print::out("Doing an allgather to get", comm_size, "ranks.");
    // Get our "cxi address"
    static constexpr size_t name_array_len = 100;
    char                    name[name_array_len];
    size_t                  _array_size = name_array_len;
    check_libfabric(fi_getname(&(ep)->fid, name, &_array_size));

    // All other ranks
    char* all_names = (char*)calloc(_array_size * comm_size, sizeof(char));
    force_mpi(MPI_Allgather(name, _array_size, MPI_CHAR, all_names, _array_size, MPI_CHAR,
                            comm));

    peers.resize(comm_size, 0);
    check_libfabric(fi_av_insert(av, all_names, comm_size, peers.data(), 0, 0));

    free(all_names);
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
        converted_request = std::make_unique<CXIRecvOneSided>(req, my_buffer, libfab,
                                                              libfab.get_peer(my_rank));
    }
    else if (Communication::Operation::RSEND == req.operation)
    {
        converted_request =
            std::make_unique<CXIRSend>(req, my_buffer, libfab, libfab.get_peer(my_rank));
    }
    else if (Communication::Operation::SEND == req.operation)
    {
        converted_request =
            std::make_unique<CXISend>(req, my_buffer, libfab, libfab.get_peer(my_rank));
    }
    else
    {
        throw std::runtime_error("Operation not supported");
    }

    converted_request->match(match_phase_a, match_phase_b);

    request_map.insert(std::make_pair(req.getID(), std::move(converted_request)));
}

LibfabricInstance::~LibfabricInstance()
{
    check_libfabric(fi_close(&(ep)->fid));
    check_libfabric(fi_close(&(progress_ctr)->fid));
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

__global__ void add_to_counters(GPUTriggers triggers)
{
    for (int x = 0; x < triggers.actual_size; x++)
    {
        // Only support increasing by one for now
        *(triggers.buffer[x]) = 1;
    }
}

__global__ void wait_on_completion(volatile size_t* comp_addr, size_t goal_value)
{
    size_t curr_value = *comp_addr;
    while (curr_value < goal_value)
    {
        curr_value = *comp_addr;
    }
}

__global__ void wait_on_completions(GPUCompletions completions)
{
    for (int x = 0; x < completions.actual_size; x++)
    {
        size_t curr_value = *(completions.buffer[x].comp_addr);
        while (curr_value < (completions.buffer[x].goal_value))
        {
            curr_value = *(completions.buffer[x].comp_addr);
        }
    }
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

void CXIQueue::enqueue_startall(std::vector<std::shared_ptr<Request>> requests)
{
#if defined(USE_TIOGA)
    /* If any kernel needs a flush, go ahead and do it first */
    for (auto& req : requests)
    {
        if (req->needs_gpu_flush())
        {
            flush_memory();
            break;
        }
    }
#endif

    GPUTriggers all_in_one;
    size_t      num_reqs    = 0;
    bool        should_bump = false;
    for (auto& req : requests)
    {
        if (Communication::Operation::BARRIER <= req->operation)
        {
            HIPQueue::enqueue_operation(req);
        }
        else
        {
            CXIObjects& cxi_req = request_map.at(req->getID());
            /* Start Request */
            Print::out("Staring request:", req->getID());
            TriggerStatus actions = cxi_req->start_cpu(*the_gpu_counter);
            Print::out("Request request:", actions);

            /* Add request to be waited on. */
            active_requests.push_back(req->getID());

            /* Enqueue global counter bump, if needed */
            if (!should_bump && (TriggerStatus::GLOBAL_BUMP == actions))
            {
                should_bump                 = true;
                all_in_one.buffer[num_reqs] = (uint64_t*)(the_gpu_counter->gpu_mmio_addr);
                num_reqs++;
            }
            if (TriggerStatus::EXTRA_BUMP == actions)
            {
                all_in_one.buffer[num_reqs] = cxi_req->get_gpu_trigger();
                num_reqs++;
            }

            if (TriggerStatus::NOT_NEEDED != actions &&
                (0 == (num_reqs % CompletionBufferFactory::MAX_GPU_COMPLETIONS)))
            {
                all_in_one.actual_size = num_reqs;
                add_to_counters<<<1, 1, 0, *my_stream>>>(all_in_one);
                force_kernel();
                num_reqs = 0;
            }
        }
    }

    if(should_bump)
    {
        /* If we used the global counter, increase its use_count */
        the_gpu_counter->up_use_count();
    }

    all_in_one.actual_size = num_reqs;
    add_to_counters<<<1, 1, 0, *my_stream>>>(all_in_one);
    force_kernel();
    Print::out("Done staring all requests.");
}

void CXIQueue::enqueue_waitall()
{
    GPUCompletions all_in_one;
    size_t         num_reqs = 0;
    for (auto req : active_requests)
    {
        all_in_one.buffer[num_reqs] = request_map.at(req)->get_gpu_completion();
        Print::out("Waiting on request:", req, all_in_one.buffer[num_reqs].comp_addr,
                   all_in_one.buffer[num_reqs].goal_value);
        num_reqs++;
        if (0 == (num_reqs % CompletionBufferFactory::MAX_GPU_COMPLETIONS))
        {
            all_in_one.actual_size = num_reqs;
            wait_on_completions<<<1, 1, 0, *my_stream>>>(all_in_one);
            force_kernel();
            num_reqs = 0;
        }
    }
    all_in_one.actual_size = num_reqs;
    wait_on_completions<<<1, 1, 0, *my_stream>>>(all_in_one);
    force_kernel();
    active_requests.clear();
    // If there are any requests offloaded to CPU
    if (entries.size() > 0)
    {
        HIPQueue::enqueue_waitall();
    }
}

void CXIQueue::flush_memory()
{
    Print::out("<E> Enqueuing buffer_wbl2 kernel");
    flush_buffer<<<1, 1, 0, *my_stream>>>();
}
