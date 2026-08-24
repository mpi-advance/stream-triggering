#ifndef ST_CXI_QUEUE_REQUEST_PROTOCOLS
#define ST_CXI_QUEUE_REQUEST_PROTOCOLS

#include "queues/cxi/requests/request_base.hpp"

class CXIRSend : public CXIRequest
{
public:
    CXIRSend(Communication::Request& user_request, CompletionBufferFactory& buffers,
             LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRequest(user_request, buffers),
          work_entry(_libfab.ep,
                     {user_request.send_buffer,
                      static_cast<size_t>(user_request.get_size_of_buffer())},
                     _libfab.get_peer(user_request.cw_peer)),
          local_completion(_libfab.ep, self, completion_buffer.get_rma_ioc_addr()),
          libfab(_libfab),
          completion_a(_libfab.alloc_counter(true)),
          completion_b(_libfab.alloc_counter(true))
    {
        work_entry.set_completion_counter(completion_a);
        local_completion.set_trigger_counter(completion_a);
        local_completion.set_completion_counter(completion_b);
    }

    ~CXIRSend()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_b);
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_rndv(work_entry.get_rma_iov_addr(),
                                     protocol_buffer.get_rma_ioc_addr(), base_req, comm_a,
                                     comm_b);
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        // Update threshold of chained things
        work_entry.set_threshold(trigger_cntr.get_next_value());
        // Adjust the triggering counter to use
        work_entry.set_trigger_counter(trigger_cntr);

        // Queue up send of user data
        // work_entry.print();
        libfab.queue_work(work_entry.get_dwqe());

        // Queue up completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        return TriggerStatus::GLOBAL_BUMP;
    }

protected:
    // Structs for the DFWQ Entry:
    RMAEntry    work_entry;
    AtomicEntry local_completion;

    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    struct fid_cntr* completion_a;
    struct fid_cntr* completion_b;
};

class CXISend : public CXIRSend
{
public:
    CXISend(Communication::Request& user_request, CompletionBufferFactory& buffers,
            LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRSend(user_request, buffers, _libfab, self), triggered(_libfab)
    {
        work_entry.set_trigger_counter(triggered);

        /* Setup CTS buffer */
        force_gpu(hipHostMalloc(&cts_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE,
                                hipHostMallocDefault));
        cts_mr = _libfab.create_mr_with_counter(
            cts_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE, FI_REMOTE_WRITE,
            FI_MR_ALLOCATED | FI_RMA_EVENT, triggered.counter, FI_REMOTE_WRITE);

        /* Set Protocol Buffer */
        protocol_buffer =
            CompletionBuffer(cts_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE, 1,
                             fi_mr_key(cts_mr), 0);
    }

    ~CXISend()
    {
        /* Free CTS related items */
        check_gpu(hipHostFree(cts_buffer));
        check_libfabric(fi_close(&(cts_mr)->fid));
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        // Update threshold of chained things
        work_entry.set_threshold(triggered.get_next_value() * 2);

        // Queue up send of user data
        // work_entry.print();
        libfab.queue_work(work_entry.get_dwqe());

        // Queue up completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        // triggered.enqueue_trigger(the_stream);
        return TriggerStatus::EXTRA_BUMP;
    }

    GPUTriggerDescription get_gpu_trigger() override
    {
        triggered.up_use_count();
        return (uint64_t*)(triggered.gpu_mmio_addr);
    }

private:
    CXICounter triggered;
    void*      cts_buffer;
    fid_mr*    cts_mr;
};

class CXIRSendShared : public CXIRequest
{
public:
    CXIRSendShared(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers),
          peer_buffer_ptr(nullptr),
          peer_completion_ptr(nullptr),
          first_time(true)
    {
    }

    ~CXIRSendShared()
    {
        if (!first_time)
        {
            check_gpu(hipIpcCloseMemHandle(peer_buffer_ptr));
            check_gpu(hipIpcCloseMemHandle(peer_completion_ptr));
        }
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_hip_ipc(ipc_data, base_req, comm_a);
    }

protected:
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        if (first_time)
        {
            base_req.out("Opening handles for send request!");
            force_gpu(hipIpcOpenMemHandle(&peer_buffer_ptr, ipc_data[0].handle,
                                          hipIpcMemLazyEnablePeerAccess));
            force_gpu(hipIpcOpenMemHandle(&peer_completion_ptr, ipc_data[1].handle,
                                          hipIpcMemLazyEnablePeerAccess));
            first_time = false;
        }
        Print::out("<E> Using Offsets for IPC:", ipc_data[0].offset, ipc_data[1].offset,
                   base_req.get_size_of_buffer(), peer_buffer_ptr);

        void* peer_true_buffer = (char*)peer_buffer_ptr + ipc_data[0].offset;
        force_gpu(hipMemcpyDtoDAsync(peer_true_buffer, base_req.send_buffer,
                                     base_req.get_size_of_buffer(), *the_stream));

        void* peer_true_completion = (char*)peer_completion_ptr + ipc_data[1].offset;
        force_gpu(hipMemcpyHtoDAsync(peer_true_completion, &num_times_started,
                                     sizeof(num_times_started), *the_stream));

        return TriggerStatus::DONE;
    }

private:
    void* peer_buffer_ptr;
    void* peer_completion_ptr;
    bool  first_time;
    // [0] = remote buffer, [1] = remote completion
    std::array<Match::Protocol::IPCBundle, 2> ipc_data;
};

class CXIRSendSelf : public CXIRequest
{
public:
    CXIRSendSelf(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_self(remote_data, base_req, comm_a, comm_b);
    }

protected:
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        Print::out("<E> Self send wants to go to:", remote_data[0], remote_data[1]);

        force_gpu(hipMemcpyDtoDAsync(remote_data[0], base_req.send_buffer,
                                     base_req.get_size_of_buffer(), *the_stream));

        force_gpu(hipMemcpyHtoDAsync(remote_data[1], &num_times_started,
                                     sizeof(num_times_started), *the_stream));

        return TriggerStatus::DONE;
    }

private:
    // [0] = remote buffer, [1] = remote completion
    Match::Protocol::SelfBundle remote_data;
};

class CXISendCredit : public CXIRSend
{
public:
    CXISendCredit(Communication::Request& user_request, CompletionBufferFactory& buffers,
                  LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRSend(user_request, buffers, _libfab, self)
    {
        credit_slots.reserve(Communication::MAX_CREDIT_SLACK);
        for (size_t index = 0; index < Communication::MAX_CREDIT_SLACK; index++)
        {
            CXICounter temp_credit_counter(_libfab);
            void*      temp_credit_buffer;
            force_gpu(hipHostMalloc(&temp_credit_buffer,
                                    CompletionBufferFactory::DEFAULT_ITEM_SIZE,
                                    hipHostMallocDefault));
            fid_mr* temp_credit_mr = _libfab.create_mr_with_counter(
                temp_credit_buffer, CompletionBufferFactory::DEFAULT_ITEM_SIZE,
                FI_REMOTE_WRITE, FI_MR_ALLOCATED | FI_RMA_EVENT,
                temp_credit_counter.counter, FI_REMOTE_WRITE);
            credit_slots.emplace_back(std::move(temp_credit_counter), temp_credit_buffer,
                                      temp_credit_mr);
            Print::out("CREDIT MR SLOT:", temp_credit_buffer,
                       temp_credit_counter.gpu_mmio_addr, temp_credit_mr);
            credit_buffers[index] = {0, 1, fi_mr_key(temp_credit_mr)};
        }
    }

    ~CXISendCredit()
    {
        for (auto& slot : credit_slots)
        {
            force_gpu(hipHostFree(std::get<1>(slot)));
            check_libfabric(fi_close(&(std::get<2>(slot))->fid));
        }
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_credit(&credit_buffer_details, credit_buffers, base_req,
                                       comm_a, comm_b);
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        size_t mr_counter_index =
            (num_times_started - 1) % Communication::MAX_CREDIT_SLACK;
        size_t threshold =
            ((num_times_started - 1) / Communication::MAX_CREDIT_SLACK) * 2 + 1;

        /* Update work entry to the correct remote buffer data */
        auto     user_buffer_size  = base_req.get_size_of_buffer();
        uint64_t offset            = mr_counter_index * user_buffer_size;
        credit_buffer_details.addr = offset;
        credit_buffer_details.len  = user_buffer_size;
        Print::out("Using:", credit_buffer_details.addr, credit_buffer_details.len,
                   credit_buffer_details.key);
        work_entry.set_rma_iov(credit_buffer_details);
        /* Update work entry to use current credit threshold */
        work_entry.set_threshold(threshold);
        /* Adjust the triggering counter to use the credit counter */
        work_entry.set_trigger_counter(std::get<0>(credit_slots.at(mr_counter_index)));

        // Queue up send of user data
        // work_entry.print();
        libfab.queue_work(work_entry.get_dwqe());

        // Queue up completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        return TriggerStatus::EXTRA_BUMP;
    }

    GPUTriggerDescription get_gpu_trigger() override
    {
        size_t mr_counter_index =
            (num_times_started - 1) % Communication::MAX_CREDIT_SLACK;
        auto& triggered = std::get<0>(credit_slots.at(mr_counter_index));
        triggered.up_use_count();
        return (uint64_t*)(triggered.gpu_mmio_addr);
    }

private:
    std::vector<std::tuple<CXICounter, void*, fid_mr*>> credit_slots;
    struct fi_rma_iov                                   credit_buffer_details;
    Match::Protocol::CreditArray                        credit_buffers;
};

template <bool USE_EAGER>
class CXIRecvOneSided : public CXIRequest
{
public:
    CXIRecvOneSided(Communication::Request&  user_request,
                    CompletionBufferFactory& buffers, LibfabricInstance& _libfab,
                    fi_addr_t self)
        : CXIRequest(user_request, buffers),
          libfab(_libfab),
          cts_entry(_libfab.ep, _libfab.get_peer(user_request.cw_peer)),
          completion_a(_libfab.alloc_counter(true)),   // CTS DWQ completion tracker
          completion_b(_libfab.alloc_counter(false)),  // registered with user MR
          completion_c(_libfab.alloc_counter(true)),   // Local completion DWQ Tracker
          local_completion(_libfab.ep, self, completion_buffer.get_rma_ioc_addr())
    {
        my_mr = _libfab.create_mr_with_counter(
            user_request.recv_buffer, user_request.get_size_of_buffer(), FI_REMOTE_WRITE,
            FI_MR_ALLOCATED | FI_RMA_EVENT, completion_b, FI_REMOTE_WRITE);

        user_buffer_rma_iov = {0, user_request.get_size_of_buffer(), fi_mr_key(my_mr)};

        cts_entry.set_completion_counter(completion_a);
        local_completion.set_trigger_counter(completion_b);
        local_completion.set_completion_counter(completion_c);
    }

    ~CXIRecvOneSided()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_c);
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
        libfab.dealloc_counter(completion_b);
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::receiver_rndv(&user_buffer_rma_iov, cts_entry.get_rma_ioc_addr(),
                                       base_req, comm_a, comm_b);
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        TriggerStatus rc = TriggerStatus::NOT_NEEDED;
        if constexpr (!USE_EAGER)
        {
            Print::out("Queue CTS to Libfabric!");
            // Update threshold of chained things
            cts_entry.set_threshold(trigger_cntr.get_next_value());
            // Adjust the triggering counter to use
            cts_entry.set_trigger_counter(trigger_cntr);
            libfab.queue_work(cts_entry.get_dwqe());
            rc = TriggerStatus::GLOBAL_BUMP;
        }

        // Queue up completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        return rc;
    }

private:
    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    // CTS Preparations
    struct fid_cntr* completion_a;
    AtomicEntry      cts_entry;

    // Allocated Libfabric Objects
    struct fid_mr* my_mr;

    // User buffer details
    struct fi_rma_iov user_buffer_rma_iov;

    // Local completion
    struct fid_cntr* completion_b;
    struct fid_cntr* completion_c;
    AtomicEntry      local_completion;
};

class CXIRecvShared : public CXIRequest
{
public:
    CXIRecvShared(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
        /* Fill in ipc_data with blanks for user data, but real completion buffer IPC
         * handle and offset */
        ipc_data = {
            {{0, 0}, {buffers.get_ipc_handle(), completion_buffer.iov_description.addr}}};
        /* Get the IPC handle of user's buffer */
        force_gpu(hipIpcGetMemHandle(&(ipc_data[0].handle), user_request.recv_buffer));
        /* Figure out user buffers' original allocation, and offset into it. */
        void*  original_ptr;
        size_t original_size;
        force_gpu(hipMemGetAddressRange(&original_ptr, &original_size,
                                        user_request.recv_buffer));
        ipc_data[0].offset = (char*)user_request.recv_buffer - (char*)original_ptr;
        Print::out("IPC User Offset:", ipc_data[0].offset,
                   "IPC Comp Offset:", ipc_data[1].offset);
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::receiver_hip_ipc(ipc_data, base_req, comm_a);
    }

protected:
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        return TriggerStatus::NOT_NEEDED;
    }

private:
    // [0] = buffer, [1] = completion
    std::array<Match::Protocol::IPCBundle, 2> ipc_data;
};

class CXIRecvSelf : public CXIRequest
{
public:
    CXIRecvSelf(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
        self_data[0] = user_request.recv_buffer;
        self_data[1] = completion_buffer.address;
        Print::out("Self wants to go to:", self_data[0], self_data[1]);
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        Match::Protocol::receiver_self(self_data, base_req, comm_a, comm_b);
    }

protected:
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        return TriggerStatus::NOT_NEEDED;
    }

private:
    // [0] = buffer, [1] = completion
    Match::Protocol::SelfBundle self_data;
};

class CXIRecvCredit : public CXIRequest
{
public:
    CXIRecvCredit(Communication::Request& user_request, CompletionBufferFactory& buffers,
                  LibfabricInstance& _libfab, fi_addr_t self)

        : CXIRequest(user_request, buffers),
          libfab(_libfab),
          triggered(_libfab),
          credit_entry(_libfab.ep, _libfab.get_peer(user_request.cw_peer)),
          completion_a(_libfab.alloc_counter(true)),   // CTS DWQ completion tracker
          completion_b(_libfab.alloc_counter(false)),  // registered with user MR
          completion_c(_libfab.alloc_counter(true)),   // Local completion DWQ Tracker
          local_completion(_libfab.ep, self, completion_buffer.get_rma_ioc_addr())
    {
        auto buffer_size =
            user_request.get_size_of_buffer() * Communication::MAX_CREDIT_SLACK;
        force_gpu(hipMalloc(&credit_buffers, buffer_size));
        my_mr = _libfab.create_mr_with_counter(
            credit_buffers, buffer_size, FI_REMOTE_WRITE, FI_MR_ALLOCATED | FI_RMA_EVENT,
            completion_b, FI_REMOTE_WRITE);

        credit_buffer_rma_iov = {0, buffer_size, fi_mr_key(my_mr)};

        // Adjust the triggering counter to use
        credit_entry.set_trigger_counter(triggered);
        credit_entry.set_completion_counter(completion_a);
        local_completion.set_trigger_counter(completion_b);
        local_completion.set_completion_counter(completion_c);
    }

    ~CXIRecvCredit()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_c);
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
        libfab.dealloc_counter(completion_b);
        check_gpu(hipFree(credit_buffers));
    }

    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::receiver_credit(&credit_buffer_rma_iov, remote_credit_buffers,
                                         base_req, comm_a, comm_b);
    }

    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        Print::out("Queue Credit to Libfabric!");
        // Update which credit slot we want to write to
        credit_entry.set_rma_ioc(remote_credit_buffers.at(
            ((num_times_started - 1) % Communication::MAX_CREDIT_SLACK)));
        // Update threshold of chained things
        credit_entry.set_threshold(triggered.get_next_value());
        // Queue up remote credit DWQ
        libfab.queue_work(credit_entry.get_dwqe());

        // Queue up local completion DWQ
        local_completion.bump_threshold();
        libfab.queue_work(local_completion.get_dwqe());

        return TriggerStatus::NOT_NEEDED;
    }

    WaitStatus wait_gpu(hipStream_t* the_stream) override;

private:
    void* credit_buffers;

    // Reference to global libfabric stuff
    LibfabricInstance& libfab;

    // Counter trigger for credit counter
    CXICounter triggered;
    // Remote locations to send credit to
    Match::Protocol::CreditArray remote_credit_buffers;

    // Credit DWQ Preparations
    struct fid_cntr* completion_a;
    AtomicEntry      credit_entry;

    // Allocated Libfabric Objects
    struct fid_mr* my_mr;

    // User buffer details
    struct fi_rma_iov credit_buffer_rma_iov;

    // Local completion
    struct fid_cntr* completion_b;
    struct fid_cntr* completion_c;
    AtomicEntry      local_completion;
};

#endif
