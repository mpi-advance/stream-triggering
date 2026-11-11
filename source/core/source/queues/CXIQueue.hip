#include "abstract/match.hpp"
#include "queues/cxi/CXIQueue.hpp"
#include "queues/cxi/libfabric_wrappers.hpp"
#include "queues/cxi/requests/request_protocols.hpp"
#include "safety/gpu.hpp"
#include "safety/libfabric.hpp"

/**
 * @file CXIQueue.hip
 * @brief Implementation for various methods and GPU kernels used by the CXIQueue.
 */

void LibfabricInstance::initialize_libfabric(MPI_Comm comm)
{
    int comm_size = -1;
    force_mpi(MPI_Comm_size(comm, &comm_size));

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
    check_libfabric(fi_ep_bind(ep, &(progress_ctr)->fid, FI_RECV | FI_WRITE));
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
    int comm_size = -1;
    force_mpi(MPI_Comm_size(comm, &comm_size));
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

void CXIQueue::exchange_protocol(Request& req)
{
    bool same_node = (MPI_UNDEFINED != Communication::Request::rankLookup(
                                           req.peer, req.comm, on_node_peers));

    // Forward looking lookup for when "comm_base" is not MPI_COMM_WORLD.
    // However, all of the Match::Protocol classes use MPI_COMM_WORLD ids.
    bool to_self =
        (my_rank == Communication::Request::rankLookup(req.peer, req.comm, comm_base));

    req.out("On node:", same_node, "Self:", to_self);

    if ((Communication::Operation::RSEND == req.operation) && to_self)
    {
        req.protocol = Communication::Protocol::SELF;
    }
    else if (Communication::IPC_PROTOCOL_ENABLED &&
             (Communication::Operation::RSEND == req.operation) && same_node)
    {
        req.protocol = Communication::Protocol::IPC;
    }
    else if ((Communication::Operation::RSEND == req.operation))
    {
        req.protocol = Communication::Protocol::EAGER;
    }
    else if (Communication::CREDIT_PROTOCOL_ENABLED &&
             (Communication::Operation::SEND == req.operation) &&
             (Communication::CREDIT_SIZE_CUTOFF >= req.get_size_of_buffer()))
    {
        req.protocol = Communication::Protocol::CREDIT;
    }
    else if ((Communication::Operation::SEND == req.operation))
    {
        req.protocol = Communication::Protocol::RNDV;
    }
    else
    {
        /* RECV will get the protocol from the match. Not technically needed since it is
         * defaulted to none */
        req.protocol = Communication::Protocol::NONE;
    }

    // Separate communicator to avoid matching conflicts.
    req.start_protocol_exchange();
}

void CXIQueue::prepare_cxi_mr_key(Request& req)
{
    /* Make sure request has the protocol exchange done */
    req.wait_on_protocol();

    /* New request that prepares the user's request to be enqueued later */
    std::unique_ptr<CXIRequest> cxi_request = nullptr;

    /* Create the object that will carry out the protocol for the operation*/
    if (Communication::Operation::RECV == req.operation)
    {
        switch (req.protocol)
        {
            case Communication::Protocol::IPC:
                cxi_request = std::make_unique<CXIRecvShared>(req, my_buffer);
                break;
            case Communication::Protocol::EAGER:
                cxi_request = std::make_unique<CXIRecvEager>(req, my_buffer, libfab,
                                                             libfab.get_peer(my_rank));
                break;
            case Communication::Protocol::CREDIT:
                cxi_request = std::make_unique<CXIRecvCredit>(req, my_buffer, libfab,
                                                              libfab.get_peer(my_rank));
                break;
            case Communication::Protocol::RNDV:
                cxi_request = std::make_unique<CXIRecvRndv>(req, my_buffer, libfab,
                                                            libfab.get_peer(my_rank));
                break;
            case Communication::Protocol::SELF:
                cxi_request = std::make_unique<CXIRecvSelf>(req, my_buffer);
                break;
            default:
                cxi_request = nullptr;
                break;
        }
    }
    else if (Communication::Operation::RSEND == req.operation)
    {
        switch (req.protocol)
        {
            case Communication::Protocol::IPC:
                cxi_request = std::make_unique<CXIRSendShared>(req, my_buffer);
                break;
            case Communication::Protocol::EAGER:
                cxi_request = std::make_unique<CXIRSend>(req, my_buffer, libfab,
                                                         libfab.get_peer(my_rank));
                break;
            case Communication::Protocol::SELF:
                cxi_request = std::make_unique<CXIRSendSelf>(req, my_buffer);
                break;
            default:
                cxi_request = nullptr;
                break;
        }
    }
    else if (Communication::Operation::SEND == req.operation)
    {
        switch (req.protocol)
        {
            case Communication::Protocol::CREDIT:
                cxi_request = std::make_unique<CXISendCredit>(req, my_buffer, libfab,
                                                              libfab.get_peer(my_rank));
                break;
            case Communication::Protocol::RNDV:
                cxi_request = std::make_unique<CXISend>(req, my_buffer, libfab,
                                                        libfab.get_peer(my_rank));
                break;
            default:
                cxi_request = nullptr;
                break;
        }
    }

    if (nullptr == cxi_request)
    {
        req.out("No action for:", req.operation, req.protocol);
        throw std::runtime_error(
            "Unable to determine actions for Operation and Protocol.");
    }

    cxi_request->match(match_phase_a, match_phase_b);

    request_map.insert(std::make_pair(req.getID(), std::move(cxi_request)));
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

/**
 * @brief A GPU kernel for triggering a counter by adding a value to it.
 * @details Note that to add to the counter, we have to assign a value to the MMIO
 * address.
 * @param [in, out] cntr The MMIO address of the NIC counter to add to.
 * @param [in] value The value to add to the NIC counter.
 */
__global__ void add_to_counter(uint64_t* cntr, size_t value)
{
    *cntr = value;
}

/**
 * @brief A GPU kernel for triggering multiple counters in a single kernel.
 * @details Currently this function only supports triggering by adding "1" to each
 * counter (via assignment). The kernel also assumes that the object passed will fit in
 * the kernel-argument space of a GPU (hence why it is not a pointer type).
 * @param [in] triggers A struct containing an array of all of the addresses to trigger.
 * @sa GPUTriggers
 * @sa CompletionBufferFactory::MAX_GPU_COMPLETIONS
 */
__global__ void add_to_counters(GPUTriggers triggers)
{
    for (int x = 0; x < triggers.actual_size; x++)
    {
        // Only support increasing by one for now
        *(triggers.buffer[x]) = 1;
    }
}

/**
 * @brief Spin wait on a memory address until it hits a specified value.
 * @details Unlike ::add_to_counter(), the address passed to this function can be any GPU
 * memory (as long as it can update while the GPU is reading it). Generally launched as a
 * <<<1,1>>> kernel.
 * @param [in] comp_addr The address of the value to spin on.
 * @param [in] goal_value The goal value to reach.
 */
__global__ void wait_on_completion(volatile size_t* comp_addr, size_t goal_value)
{
    size_t curr_value = *comp_addr;
    while (curr_value < goal_value)
    {
        curr_value = *comp_addr;
    }
}

/**
 * @brief Spin wait on multiple memory addresses until each hits a specified value.
 * @details Each address is checked sequentially, and each address can have its own
 * completion value. Like ::add_to_counters(), the parameter to this function is also
 * assumed to fit in the kernel-argument space of a GPU.
 * @param [in] completions A struct containing an array of completions to work over.
 * @sa GPUCompletions
 */
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

/**
 * @brief [TODO]
 * @param [in] completions
 * @param [in] copies
 */
__global__ void wait_and_copy(GPUCompletions completions, GPUCreditCopies copies)
{
    const size_t index       = threadIdx.x;
    const size_t num_threads = blockDim.x;

    if (0 == index)
    {
        for (size_t x = 0; x < completions.actual_size; x++)
        {
            while (*(completions.buffer[x].comp_addr) < completions.buffer[x].goal_value)
            {
            }
        }
    }
    __syncthreads();

    for (size_t c = 0; c < copies.actual_size; c++)
    {
        const double4* __restrict__ c_temp =
            reinterpret_cast<const double4*>(copies.buffer[c].temp_buffer);
        double4* __restrict__ c_user =
            reinterpret_cast<double4*>(copies.buffer[c].user_buffer);
        const size_t size               = copies.buffer[c].size;
        const size_t max_complete_index = size / sizeof(double4);
        for (size_t i = index; i < max_complete_index; i += num_threads)
        {
            c_user[i] = c_temp[i];
        }
        const size_t remainder = size % sizeof(double4);
        if (index < remainder)
        {
            reinterpret_cast<char*>(&c_user[max_complete_index])[index] =
                reinterpret_cast<const char*>(&c_temp[max_complete_index])[index];
        }
    }

    /* Every thread is done with every staging slot before the credits go back */
    __syncthreads();
    if (0 == index)
    {
        for (size_t c = 0; c < copies.actual_size; c++)
        {
            *(copies.buffer[c].cntr) = 1;
        }
    }
}

/** @brief A kernel specific to MI250X GPUs for flushing the GPU L2 buffer cache.
 * @details This kernel is used by CXIQueue::flush_memory() (which is used when enqueueing
 * the starts and waits of communication operations) to force the GPU to update
 * coarse-grained GPU memory without needing intervention from the host (i.e.,
 * hipDeviceSynchronize or hipStreamSynchronize). It directly invokes ROCm assembly.
 * This function is only enable when CMake detects the MI250x (and adds the "USE_TIOGA"
 * compiler definition)
 */
__global__ void flush_buffer()
{
    asm volatile("buffer_wbl2");
}

WaitStatus CXIRequest::wait_gpu(hipStream_t* the_stream)
{
    return WaitStatus::GLOBAL_KERNEL;
}

WaitStatus CXIRecvCredit::wait_gpu(hipStream_t* the_stream)
{
    /* The credit goes back from the waitall's wait_and_copy kernel */
    credit_trigger_cntr.up_use_count();
    return WaitStatus::CREDIT_FUSED;
}

GPUCreditCopy CXIRecvCredit::get_credit_copy()
{
    auto buffer_size = base_req.get_size_of_buffer();
    return {
        (char*)staging_buffers() +
            (((num_times_started - 1) % Communication::MAX_CREDIT_SLACK) * buffer_size),
        base_req.recv_buffer, buffer_size,
        (uint64_t*)(credit_trigger_cntr.gpu_mmio_addr)};
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

    GPUTriggers                           all_in_one;
    size_t                                num_reqs    = 0;
    bool                                  should_bump = false;
    std::vector<std::shared_ptr<Request>> collectives;
    for (auto& req : requests)
    {
        if (Communication::Operation::BARRIER <= req->operation)
        {
            collectives.push_back(req);
        }
        else
        {
            CXIObjects& cxi_req = request_map.at(req->getID());
            /* Start Request */
            Print::out("Starting request:", req->getID());
            TriggerStatus actions = cxi_req->start_cpu(*the_gpu_counter, my_stream);
            Print::out("Request action:", (int)actions);

            /* If request does not need to be waited on, continue */
            if (TriggerStatus::DONE == actions)
            {
                continue;
            }

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

            if (CompletionBufferFactory::MAX_GPU_COMPLETIONS == num_reqs)
            {
                all_in_one.actual_size = num_reqs;
                Print::out("<E> Launching add_to_counters for", num_reqs, "requests.");
                add_to_counters<<<1, 1, 0, *my_stream>>>(all_in_one);
                force_kernel();
                num_reqs = 0;
            }
        }
    }

    if (should_bump)
    {
        /* If we used the global counter, increase its use_count */
        the_gpu_counter->up_use_count();
    }

    if (num_reqs > 0)
    {
        all_in_one.actual_size = num_reqs;
        Print::out("<E> Launching add_to_counters for", num_reqs, "requests.");
        add_to_counters<<<1, 1, 0, *my_stream>>>(all_in_one);
        force_kernel();
    }

    if (!collectives.empty())
    {
        HIPQueue::enqueue_startall(collectives);
    }
}

void CXIQueue::enqueue_waitall(std::vector<std::shared_ptr<Request>> requests,
                               MPI_Status*                           statuses)
{
#if defined(USE_TIOGA)
    flush_memory();  // TODO: Refine this more so not all waitalls need this
#endif
    GPUCompletions all_in_one;
    size_t         num_reqs        = 0;
    bool           has_collectives = false;
    /* CREDIT receives are copied, and their credit returned, by the wait kernel */
    GPUCreditCopies copies;
    copies.actual_size = 0;
    auto launch_waits  = [&](const char* which) {
        all_in_one.actual_size = num_reqs;
        if (0 == copies.actual_size)
        {
            Print::out("<E> Enqueuing wait_on_completions kernel", which);
            wait_on_completions<<<1, 1, 0, *my_stream>>>(all_in_one);
        }
        else
        {
            Print::out("<E> Enqueuing wait_and_copy kernel", which);
            wait_and_copy<<<1, 1024, 0, *my_stream>>>(all_in_one, copies);
        }
        force_kernel();
        num_reqs           = 0;
        copies.actual_size = 0;
    };
    for (auto& req : requests)
    {
        if (Communication::Operation::BARRIER <= req->operation)
        {
            has_collectives = true;
            continue;
        }
        auto found = request_map.find(req->getID());
        if (request_map.end() == found || !found->second->take_outstanding())
        {
            continue;
        }
        const WaitStatus wait = found->second->wait_gpu(my_stream);
        if (WaitStatus::GLOBAL_KERNEL == wait || WaitStatus::CREDIT_FUSED == wait)
        {
            all_in_one.buffer[num_reqs] = found->second->get_gpu_completion();
            Print::out("Waiting on request:", req->getID(),
                       all_in_one.buffer[num_reqs].comp_addr,
                       all_in_one.buffer[num_reqs].goal_value);
            num_reqs++;
            if (WaitStatus::CREDIT_FUSED == wait)
            {
                copies.buffer[copies.actual_size++] = found->second->get_credit_copy();
            }
            if (CompletionBufferFactory::MAX_GPU_COMPLETIONS == num_reqs ||
                GPUCreditCopies::MAX_COPIES == copies.actual_size)
            {
                launch_waits("A");
            }
        }
    }

    if (num_reqs > 0)
    {
        launch_waits("B");
    }

    /* Collectives are offloaded to the CPU; point-to-point positions get empty statuses
     */
    if (has_collectives)
    {
        HIPQueue::enqueue_waitall(requests, statuses);
    }
    else if (nullptr != statuses)
    {
        /* Skip the progress engine and stream wait: no CPU work to wait on */
        std::vector<MPI_Request> nulls(requests.size(), MPI_REQUEST_NULL);
        force_mpi(MPI_Waitall(nulls.size(), nulls.data(), statuses));
    }
}

void CXIQueue::flush_memory()
{
    Print::out("<E> Enqueuing buffer_wbl2 kernel");
    flush_buffer<<<1, 1, 0, *my_stream>>>();
}
