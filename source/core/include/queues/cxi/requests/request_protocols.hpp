#ifndef ST_CXI_QUEUE_REQUEST_PROTOCOLS
#define ST_CXI_QUEUE_REQUEST_PROTOCOLS

#include "queues/cxi/requests/request_base.hpp"

/**
 * @file request_protocols.hpp
 * @brief File for all derivatives of ::CXIRequest.
 */

/**
 * @brief Implementation of Communication::Protocol::EAGER send operations.
 * @details Aligns with Communication::Operation::RSEND operations. This method works by
 * creating two deferred work queue entries. The first, triggered by the GPU, is for
 * moving the data to the other side (in an eager manner). The second, chained off the
 * completion of the first, performs a local atomic increment on the completion buffer so
 * the GPU can watch for when the operation is done.
 *
 * Is able to be triggered off the global counter.
 */
class CXIRSend : public CXIRequest
{
public:
    /**
     * @brief Constructor for eager/rsend operations.
     * @details Takes care of setting up the two deferred work queue entries, and aligns
     * all but the initial triggering counter. Allocates two counters to use as completion
     * counters (and they are marked for checking progress on,
     * ::LibfabricInstance::progress_dwq()).
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     */
    CXIRSend(Communication::Request& user_request, CompletionBufferFactory& buffers,
             LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRequest(user_request, buffers),
          work_entry(_libfab.ep, _libfab.get_peer(user_request.cw_peer),
                     {user_request.send_buffer,
                      static_cast<size_t>(user_request.get_size_of_buffer())}),
          local_completion(_libfab.ep, self, *completion_buffer.get_rma_ioc_addr()),
          libfab(_libfab),
          completion_a(_libfab.alloc_counter(true)),
          completion_b(_libfab.alloc_counter(true))
    {
        work_entry.set_completion_counter(completion_a);
        local_completion.set_trigger_counter(completion_a);
        local_completion.set_completion_counter(completion_b);
    }

    /** @brief Destructor that deallocates the two libfabric counters created. */
    ~CXIRSend()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_b);
    }

    /** [UPDATE]
     * @brief Initiate matching between two communications.
     * @details This class calls out to ::Match::Protocol::sender_rndv(), but passes a
     * blank CXIRequest::protocol_buffer.
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @param [in] comm_b Communication::MPIS_COMM_WORLD duplicate for Phase B
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_eager(work_entry.get_rma_iov_addr(), base_req, comm_a);
    }

    /**
     * @brief Queues up two deferred work queue entries, using the global triggering
     * counter as the initial trigger.
     * @details Also bumps the threshold for the ::AtomicEntry each time it is called
     * (since this class controls that deferred work queue entry's triggering counter).
     * @param [in] trigger_cntr The global hardware counter that will be used to trigger
     * communication.
     * @param [in] the_stream The GPU stream that should be enqueued to.
     * @return ::TriggerStatus::GLOBAL_BUMP
     * @sa CXICounter::get_next_value()
     */
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
    /** @brief Deferred work queue entry for the RMA operation that moves the user
     * buffer. */
    RMAEntry work_entry;
    /** @brief Deferred work queue entry for the atomic operation that creates the local
     * completion. */
    AtomicEntry local_completion;

    /** @brief Local reference to the set of libfabric objects. */
    LibfabricInstance& libfab;

    /** @brief Allocated completion counter (for the RMA operation) */
    struct fid_cntr* completion_a;
    /** @brief Allocated completion counter (for the atomic operation) */
    struct fid_cntr* completion_b;
};

/**
 * @brief Implementation of Communication::Protocol::RNDV send operations.
 * @details Aligns with Communication::Operation::SEND operations. This method works by
 * creating two deferred work queue entries. The first, triggered by the GPU AND a signal
 * of readiness from the peer receive, is for moving the data to the other side . The
 * second, chained off the completion of the first, performs a local atomic increment on
 * the completion buffer so the GPU can watch for when the operation is done.
 *
 * Is NOT able to be triggered off the global counter.
 */
class CXISend : public CXIRSend
{
public:
    /** [REVIEW]
     * @brief Constructor for the regular send operations (using a rendezvous protocol).
     * @details This class lets the parent ::CXIRSend set up most of the deferred work
     * queue objects. Once that is done, it creates a ::CXICounter triggering counter and
     * assigns it to the first deferred work queue entry. It then allocates pinned-host
     * memory for the "clear-to-send" buffer, which will track the "ready" from the remote
     * process via a libfabric counter set up to track FI_REMOTE_WRITE actions. The peer
     * is given the memory region key during the exchange in the matching phase.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     */
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

        /* Describe CTS buffer for the peer */
        cts_details = {0, 1, fi_mr_key(cts_mr)};
    }

    /** @brief Destructor that frees the pinned-host clear-to-send buffer and its
     * libfabric memory region. */
    ~CXISend()
    {
        /* Free CTS related items */
        check_gpu(hipHostFree(cts_buffer));
        check_libfabric(fi_close(&(cts_mr)->fid));
    }

    /**
     * @brief [TODO]
     * @param [in] comm_a
     * @param [in] comm_b
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_rndv(work_entry.get_rma_iov_addr(), &cts_details,
                                     base_req, comm_a, comm_b);
    }

    /**
     * @brief Queues up two deferred work queue entries to carry out a regular send
     * operation.
     * @details Unlike ::CXIRSend, before this operation can send its data, it needs to
     * know that the other side is ready. To do this, we use the class-specific
     * ::CXICounter that has been set up to track REMOTE_WRITE operations to the
     * clear-to-send buffer (::CXISend::cts_buffer). Once one of those has happened, and
     * the local GPU has triggered the counter to be incremented, the deferred work queue
     * item for the send will be triggered. The local completion operation remains the
     * same as the parent class.
     *
     * That fact that the corresponding receive operation will block its GPU to wait on
     * the completion of the send combined with the semantic reason that it is not allowed
     * to queue two start operations on a request without a wait in between is the reason
     * why this function can get away with "triggered.get_next_value() * 2" as the
     * threshold each time.
     *
     * Also bumps the threshold for the ::AtomicEntry each time it is called
     * (since this class controls that deferred work queue entry's triggering counter).
     *
     * @param [in] trigger_cntr The global hardware counter that could be used to trigger
     * communication. Ignored in this implementation.
     * @param [in] the_stream The GPU stream that should be enqueued to.
     * @return ::TriggerStatus::EXTRA_BUMP to indicate that this operation wants a
     * specific bump on its counter.
     */
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

        return TriggerStatus::EXTRA_BUMP;
    }

    /** @brief Generate a ::GPUTriggerDescription for the operation.
     * @details Will call ::CXICounter::up_use_count() on ::CXISend::triggered.
     * @return A ::GPUTriggerDescription on how to trigger the start of the operation.
     */
    GPUTriggerDescription get_gpu_trigger() override
    {
        triggered.up_use_count();
        return (uint64_t*)(triggered.gpu_mmio_addr);
    }

private:
    /** @brief Separate counter for triggering send operations. */
    CXICounter triggered;
    /** @brief Pointer for the clear-to-send pinned-host buffer. */
    void* cts_buffer;
    /** @brief Libfabric memory region handle for the clear-to-send buffer. */
    fid_mr*           cts_mr;
    struct fi_rma_ioc cts_details;  //!< [TODO]
};

/**
 * @brief Implementation of Communication::Protocol::IPC send operations.
 * @details This class is designed as an optimization for on-node communications. As such,
 * it can leverage HIP IPC to move the data and avoid needing to go through the network
 * with libfabric.
 * While this class does exchange IPC handle information during the matching phase, the
 * remote buffers are not registered until the first start action is enqueued.
 *
 * Currently only supports "ready send" (::Communication::Operation::RSEND) operations.
 * Regular sends (::Communication::Operation::SEND) are not supported with this class at
 * the moment.
 *
 * Does not need any ::CXICounter.
 */
class CXIRSendShared : public CXIRequest
{
public:
    /**
     * @brief Constructor for on-node eager/rsend operations.
     * @details Initializes ::CXIRSendShared::first_time to true.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     */
    CXIRSendShared(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers),
          peer_buffer_ptr(nullptr),
          peer_completion_ptr(nullptr),
          first_time(true)
    {
    }

    /** @brief Destructor. Closes the IPC handles if they were opened. */
    ~CXIRSendShared()
    {
        if (!first_time)
        {
            check_gpu(hipIpcCloseMemHandle(peer_buffer_ptr));
            check_gpu(hipIpcCloseMemHandle(peer_completion_ptr));
        }
    }

    /**
     * @brief Initiate matching between two on-node, ready-based communications.
     * @details This class calls out to ::Match::Protocol::sender_hip_ipc().
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @param [in] comm_b Communication::MPIS_COMM_WORLD duplicate for Phase B
     * communication (unused).
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_hip_ipc(ipc_data, base_req, comm_a);
    }

protected:
    /**
     * @brief Carry out enqueueing the actions needed to perform an IPC-based send
     * operation.
     * @details If this is the first time the operation is to be enqueued, open the IPC
     * handles to the peer's buffers. Then move the data, and a completion signal, as two
     * separate, asynchronous HIP mem-copies on the provided HIP stream.
     *
     * This approach does not need any waiting kernels because the asynchronous
     * device mem-copies are already enqueued onto a stream by the HIP API, so when those
     * actions have completed, the data has been delivered to the other side.
     *
     * *BUG*: There seems to be a bug with HIP IPC on ROCm 6.4.X (possibly just on LLNL's
     * Tuolumne), where IPC moves over 32k require synchronization with the OTHER side
     * before the data is moved. This is fixed in ROCm 7.14.X (potentially all ROCm 7
     * versions).
     *
     * @param [in] trigger_cntr The global hardware counter that could be used to trigger
     * communication. Unused in current implementation.
     * @param [in] the_stream The GPU stream that the memory copies will be enqueued to.
     * @return ::TriggerStatus::DONE. Method does not need any sort of triggering, nor any
     * additional kernel for waiting.
     */
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
    /** @brief HIP IPC handle to remote peer's data buffer. */
    void* peer_buffer_ptr;
    /** @brief HIP IPC handle to remote peer's completion buffer. */
    void* peer_completion_ptr;
    /** @brief Have the IPC handles been opened or not. */
    bool first_time;
    /** @brief Details about the peer's buffers (as IPC handles)
     * @details Index 0 is the remote buffer and index 1 is the remote completion buffer.
     */
    std::array<Match::Protocol::IPCBundle, 2> ipc_data;
};

/**
 * @brief Implementation of Communication::Protocol::SELF for send operations.
 * @details Fairly similar to the implementation used in ::CXIRSendShared, except that
 * since this is to the same process, no IPC handles are needed. Aside from variables
 * related to the IPC handles, everything else is the same (including not supporting
 * regular send operations).
 *
 * Does not need any ::CXICounter object.
 */
class CXIRSendSelf : public CXIRequest
{
public:
    /**
     * @brief Constructor for eager/rsend operations to the same process.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     */
    CXIRSendSelf(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
    }

    /**
     * @brief Initiate matching between two on-same-process, ready-based communications.
     * @details This class calls out to ::Match::Protocol::sender_self().
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @param [in] comm_b Communication::MPIS_COMM_WORLD duplicate for Phase B
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_self(remote_data, base_req, comm_a, comm_b);
    }

protected:
    /**
     * @brief Carry out enqueueing the actions needed to perform a self-send operation.
     * @details The implementation of this is two asynchronous HIP mem-copies to the
     * addresses obtained in the matching phase. Same reasoning for the return code as
     * ::CXIRSendShared.
     * @param [in] trigger_cntr The global hardware counter that could be used to trigger
     * communication. Unused in current implementation.
     * @param [in] the_stream The GPU stream that the memory copies will be enqueued to.
     * @return ::TriggerStatus::DONE. Method does not need any sort of triggering, nor any
     * additional kernel for waiting.
     */
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
    /** @brief Details about the peer's buffers (as raw pointers)
     * @details Index 0 is the remote buffer and index 1 is the remote completion buffer.
     */
    Match::Protocol::SelfBundle remote_data;
};

/**
 * @brief Implementation of Communication::Protocol::CREDIT send operations.
 * @details This protocol attempts to combine eager and rendezvous protocols to get the
 * best of both. By pre-allocating some buffer space ahead of time, this protocol can
 * ensure that it can send a few messages in an eager manner. When these run out, it falls
 * back to needing "clear-to-send" from the peer process. The number of these credit slots
 * is controlled by ::Communication::MAX_CREDIT_SLACK. On the sending side, this value is
 * used to allocate the number of "clear-to-send" buffers. On the receiving side only a
 * single buffer and libfabric memory key are created.
 *
 * This protocol works by having the sending side essentially lower the threshold for the
 * first "credit slack" number of times so that it only depends on the trigger for the
 * first time each slot is used. After that, it depends on getting the "credit is free"
 * message back from the other process. This message is essentially a "clear-to-send" that
 * has been set up to go after the receiver has processed a message.
 *
 * Is NOT able to be triggered off the global counter.
 */
class CXISendCredit : public CXIRSend
{
public:
    /**
     * @brief Constructor for credit-based send operations. Allocates credit
     * "clear-to-send" buffers.
     * @details Each "clear-to-send" credit buffer requires a libfabric memory region and
     * counter. The buffers are allocated as host-pinned memory. Information about the
     * credit buffer slots is saved into ::CXISendCredit::credit_slots.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     */
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
            Print::out("CREDIT MR SLOT:", temp_credit_buffer,
                       temp_credit_counter.gpu_mmio_addr, temp_credit_mr);
            credit_slots.emplace_back(std::move(temp_credit_counter), temp_credit_buffer,
                                      temp_credit_mr);
            credit_buffers[index] = {0, 1, fi_mr_key(temp_credit_mr)};
        }
    }

    /** @brief Destructor that frees memory and memory regions for the credit slots. */
    ~CXISendCredit()
    {
        for (auto& slot : credit_slots)
        {
            force_gpu(hipHostFree(std::get<1>(slot)));
            check_libfabric(fi_close(&(std::get<2>(slot))->fid));
        }
    }

    /**
     * @brief Initiate matching between two credit-based communications.
     * @details This class calls out to ::Match::Protocol::sender_credit().
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @param [in] comm_b Communication::MPIS_COMM_WORLD duplicate for Phase B
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_credit(&credit_buffer_details, credit_buffers, base_req,
                                       comm_a, comm_b);
    }

    /**
     * @brief Carry out enqueueing the actions needed to perform a credit-based send
     * operation.
     * @details Each start will cycle through the remote credit buffer slots it can use.
     * The first time each slot is used, the triggering threshold only depends on the
     * sender's trigger. After that, it depends on the "credit is free" message from the
     * peer process. Beyond adjusting the remote memory region offset and the triggering
     * counter to use, this function behaves similarly to ::CXISend::start_derived().
     * @param [in] trigger_cntr The global hardware counter that could be used to trigger
     * communication. Unused in current implementation.
     * @param [in] the_stream The GPU stream that should be enqueued to.
     * @return ::TriggerStatus::EXTRA_BUMP to indicate that this operation wants a
     * specific bump on its counter.
     */
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

    /**
     * @brief Generate a ::GPUTriggerDescription for the operation.
     * @details Will call ::CXICounter::up_use_count() on one of the ::CXICounter objects
     * in ::CXISendCredit::credit_slots, according to which credit slot is in use. The
     * chosen counter is also the one whose MMIO address is returned to be triggered. Note
     * that the chosen credit slot depends on ::CXIRequest::num_times_started.
     * @return A ::GPUTriggerDescription on how to trigger the start of the operation.
     */
    GPUTriggerDescription get_gpu_trigger() override
    {
        size_t mr_counter_index =
            (num_times_started - 1) % Communication::MAX_CREDIT_SLACK;
        auto& triggered = std::get<0>(credit_slots.at(mr_counter_index));
        triggered.up_use_count();
        return (uint64_t*)(triggered.gpu_mmio_addr);
    }

private:
    /** @brief Details about the local credit "clear-to-send" buffers.
     * @details Each tuple holds the following values: Index 0 is the ::CXICounter
     * attached to the buffer; Index 1 is the raw pointer of the buffer; Index 2 is the
     * libfabric memory region of the buffer. Vector is of size
     * ::Communication::MAX_CREDIT_SLACK.
     */
    std::vector<std::tuple<CXICounter, void*, fid_mr*>> credit_slots;
    /**
     * @brief Details about the remote credit buffer.
     * @details Offset is adjusted per enqueued start operation.
     */
    struct fi_rma_iov credit_buffer_details;
    /**
     * @brief A set of fi_rma_ioc objects detailing the credit availability buffers on the
     * send side.
     * @details Only used in setup and matching phase.
     * @sa Match::Protocol::CreditBundle
     */
    Match::Protocol::CreditArray credit_buffers;
};

/**
 * @brief Implementation of Communication::Protocol::EAGER and
 * Communication::Protocol::RNDV recv operations.
 * @details Both of these protocols are fairly similar in the setup that they do,
 * including the data being matched. As such, they are combined into a single class
 * instead of separated like ::CXIRSend and ::CXISend. There is no real reason for this
 * other than this is how the class started out (the derived send classes came later).
 * @tparam USE_EAGER Determines if this object will follow Communication::Protocol::EAGER
 * or Communication::Protocol::RNDV semantics. Only affects
 * ::CXIRecvOneSided::start_derived().
 */
template <bool USE_EAGER>
class CXIRecvOneSided : public CXIRequest
{
public:
    /**
     * @brief
     * @param [in] user_request
     * @param [in, out] buffers
     * @param [in, out] _libfab
     * @param [in] self
     */
    CXIRecvOneSided(Communication::Request&  user_request,
                    CompletionBufferFactory& buffers, LibfabricInstance& _libfab,
                    fi_addr_t self)
        : CXIRequest(user_request, buffers),
          libfab(_libfab),
          cts_entry(_libfab.ep, _libfab.get_peer(user_request.cw_peer)),
          completion_a(_libfab.alloc_counter(true)),   // CTS DWQ completion tracker
          completion_b(_libfab.alloc_counter(false)),  // registered with user MR
          completion_c(_libfab.alloc_counter(true)),   // Local completion DWQ Tracker
          local_completion(_libfab.ep, self, *completion_buffer.get_rma_ioc_addr())
    {
        my_mr = _libfab.create_mr_with_counter(
            user_request.recv_buffer, user_request.get_size_of_buffer(), FI_REMOTE_WRITE,
            FI_MR_ALLOCATED | FI_RMA_EVENT, completion_b, FI_REMOTE_WRITE);

        user_buffer_rma_iov = {0, user_request.get_size_of_buffer(), fi_mr_key(my_mr)};

        cts_entry.set_completion_counter(completion_a);
        local_completion.set_trigger_counter(completion_b);
        local_completion.set_completion_counter(completion_c);
    }

    /** @brief Destructor that frees the three counters and one memory region that were
     * created. */
    ~CXIRecvOneSided()
    {
        // Free counter
        libfab.dealloc_counter(completion_a);
        libfab.dealloc_counter(completion_c);
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
        libfab.dealloc_counter(completion_b);
    }

    /** [UPDATE]
     * @brief Initiate matching between two communications.
     * @details This class calls out to ::Match::Protocol::receiver_rndv().
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @param [in] comm_b Communication::MPIS_COMM_WORLD duplicate for Phase B
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        if constexpr (USE_EAGER)
        {
            Match::Protocol::receiver_eager(&user_buffer_rma_iov, base_req, comm_a);
        }
        else
        {
            Match::Protocol::receiver_rndv(&user_buffer_rma_iov,
                                           cts_entry.get_rma_ioc_addr(), base_req, comm_a,
                                           comm_b);
        }
    }

    /**
     * @brief
     * @param [in] trigger_cntr
     * @param [in] the_stream
     * @return
     */
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
    /** @brief Local reference to the set of libfabric objects. */
    LibfabricInstance& libfab;

    /** @brief Allocated completion counter (for the clear-to-send atomic operation) */
    struct fid_cntr* completion_a;
    /** @brief Deferred work queue entry for the atomic operation that moves
     * "clear-to-send" signal to the peer. */
    AtomicEntry cts_entry;

    // Allocated Libfabric Objects
    struct fid_mr* my_mr;  //!<

    // User buffer details
    struct fi_rma_iov user_buffer_rma_iov;  //!<

    // Local completion
    struct fid_cntr* completion_b;  //!<
    /** @brief Allocated completion counter (for the Atomic operation) */
    struct fid_cntr* completion_c;
    AtomicEntry      local_completion;  //!<
};

/**
 * @brief
 * @details
 */
class CXIRecvShared : public CXIRequest
{
public:
    /**
     * @brief
     * @param [in] user_request
     * @param [in, out] buffers
     */
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

    /**
     * @brief
     * @param [in] comm_a
     * @param [in] comm_b
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::receiver_hip_ipc(ipc_data, base_req, comm_a);
    }

protected:
    /**
     * @brief
     * @param [in] trigger_cntr
     * @param [in] the_stream
     * @return
     */
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        return TriggerStatus::NOT_NEEDED;
    }

private:
    // [0] = buffer, [1] = completion
    std::array<Match::Protocol::IPCBundle, 2> ipc_data;  //!<
};

/**
 * @brief
 * @details
 */
class CXIRecvSelf : public CXIRequest
{
public:
    /**
     * @brief
     * @param [in] user_request
     * @param [in, out] buffers
     */
    CXIRecvSelf(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
        self_data[0] = user_request.recv_buffer;
        self_data[1] = completion_buffer.address;
        Print::out("Self wants to go to:", self_data[0], self_data[1]);
    }

    /**
     * @brief
     * @param [in] comm_a
     * @param [in] comm_b
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        Match::Protocol::receiver_self(self_data, base_req, comm_a, comm_b);
    }

protected:
    /**
     * @brief
     * @param [in] trigger_cntr
     * @param [in] the_stream
     * @return
     */
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        return TriggerStatus::NOT_NEEDED;
    }

private:
    // [0] = buffer, [1] = completion
    Match::Protocol::SelfBundle self_data;  //!<
};

/**
 * @brief
 * @details
 */
class CXIRecvCredit : public CXIRequest
{
public:
    /**
     * @brief
     * @param [in] user_request
     * @param [in, out] buffers
     * @param [in, out] _libfab
     * @param [in] self
     */
    CXIRecvCredit(Communication::Request& user_request, CompletionBufferFactory& buffers,
                  LibfabricInstance& _libfab, fi_addr_t self)

        : CXIRequest(user_request, buffers),
          libfab(_libfab),
          triggered(_libfab),
          credit_entry(_libfab.ep, _libfab.get_peer(user_request.cw_peer)),
          completion_a(_libfab.alloc_counter(true)),   // CTS DWQ completion tracker
          completion_b(_libfab.alloc_counter(false)),  // registered with user MR
          completion_c(_libfab.alloc_counter(true)),   // Local completion DWQ Tracker
          local_completion(_libfab.ep, self, *completion_buffer.get_rma_ioc_addr())
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

    /** @brief */
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

    /**
     * @brief
     * @param [in] comm_a
     * @param [in] comm_b
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::receiver_credit(&credit_buffer_rma_iov, remote_credit_buffers,
                                         base_req, comm_a, comm_b);
    }

    /**
     * @brief
     * @param [in] trigger_cntr
     * @param [in] the_stream
     * @return
     */
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

    /**
     * @brief
     * @param [in] the_stream
     * @return
     */
    WaitStatus wait_gpu(hipStream_t* the_stream) override;

private:
    void* credit_buffers;  //!<

    // Reference to global libfabric stuff
    LibfabricInstance& libfab;  //!<

    // Counter trigger for credit counter
    CXICounter triggered;  //!<
    // Remote locations to send credit to
    Match::Protocol::CreditArray remote_credit_buffers;  //!<

    // Credit DWQ Preparations
    struct fid_cntr* completion_a;  //!<
    AtomicEntry      credit_entry;  //!<

    // Allocated Libfabric Objects
    struct fid_mr* my_mr;  //!<

    // User buffer details
    struct fi_rma_iov credit_buffer_rma_iov;  //!<

    // Local completion
    struct fid_cntr* completion_b;      //!<
    struct fid_cntr* completion_c;      //!<
    AtomicEntry      local_completion;  //!<
};

#endif
