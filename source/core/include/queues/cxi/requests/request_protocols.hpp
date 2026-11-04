#ifndef ST_CXI_QUEUE_REQUEST_PROTOCOLS
#define ST_CXI_QUEUE_REQUEST_PROTOCOLS

#include "misc/mem_pool.hpp"
#include "queues/cxi/requests/request_base.hpp"

/**
 * @file request_protocols.hpp
 * @brief File for all derivatives of ::CXIRequest.
 */

/** @brief A base class for any ::CXIRequest derivative that needs to use a libfabric
 * deferred work queue entry to signal local completion.
 * @details Will allocate a libfabric counter (that is marked for progress tracking) and
 * uses an ::AtomicEntry to describe its work queue entry. Note that this class does not
 * setup the triggering counter for the deferred work queue entry it creates. Derived
 * classes are responsible for setting that.
 */
class CXILocalCompletion : public CXIRequest
{
public:
    /** @brief Sets up only a "local completion" deferred work queue entry and associated
     * resources (minus the triggering counter).
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     */
    CXILocalCompletion(Communication::Request&  user_request,
                       CompletionBufferFactory& buffers, LibfabricInstance& _libfab,
                       fi_addr_t self)
        : CXIRequest(user_request, buffers),
          libfab(_libfab),
          local_completion_cntr(_libfab.alloc_counter(true)),
          local_completion_entry(_libfab.ep, self, *completion_buffer.get_rma_ioc_addr())
    {
        local_completion_entry.set_completion_counter(local_completion_cntr);
    }

    /** @brief  Destructor that deallocates the allocated libfabric counter. */
    ~CXILocalCompletion()
    {
        libfab.dealloc_counter(local_completion_cntr);
    }

protected:
    /** @brief Queues up the deferred work queue entry for the local completion operation.
     * @details Derived classes should ensure that they have setup a triggering counter on
     * this local operation.
     */
    void queue_local_completion()
    {
        local_completion_entry.bump_threshold();
        libfab.queue_work(local_completion_entry.get_dwqe());
    }

    /** @brief Local reference to the set of libfabric objects. */
    LibfabricInstance& libfab;
    /** @brief Libfabric counter that is used as the completion counter for the local
     * completion operation. */
    struct fid_cntr* local_completion_cntr;
    /** @brief Deferred work queue entry for the atomic operation that creates the local
     * completion. */
    AtomicEntry local_completion_entry;
};

/** @brief Implementation of Communication::Protocol::EAGER send operations.
 * @details Aligns with Communication::Operation::RSEND operations. This method works by
 * creating two deferred work queue entries. The first, triggered by the GPU, is for
 * moving the data to the other side (in an eager manner). The second, chained off the
 * completion of the first, performs a local atomic increment on the completion buffer so
 * the GPU can watch for when the operation is done.
 *
 * Is able to be triggered off the global counter.
 */
class CXIRSend : public CXILocalCompletion
{
public:
    /** @brief Constructor for eager/rsend operations.
     * @details Takes care of setting up the two deferred work queue entries needed for
     * the Rsend operation. The second comes from its parent ::CXILocalCompletion, and the
     * first is setup in this constructor. The constructor also aligns the triggering
     * counter of the local operation to the completion of the data movement deferred work
     * queue entry. The initial triggering counter for the first deferred work queue entry
     * is not setup here. Allocates one more counter to use as a completion counter (which
     * is marked for checking progress on, ::LibfabricInstance::progress_dwq()).
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     */
    CXIRSend(Communication::Request& user_request, CompletionBufferFactory& buffers,
             LibfabricInstance& _libfab, fi_addr_t self)
        : CXILocalCompletion(user_request, buffers, _libfab, self),
          data_entry(_libfab.ep, _libfab.get_peer(user_request.cw_peer),
                     {user_request.send_buffer,
                      static_cast<size_t>(user_request.get_size_of_buffer())}),
          data_completion_cntr(_libfab.alloc_counter(true))
    {
        data_entry.set_completion_counter(data_completion_cntr);
        local_completion_entry.set_trigger_counter(data_completion_cntr);
    }

    /** @brief Destructor that deallocates the libfabric counter created. */
    ~CXIRSend()
    {
        // Free counter
        libfab.dealloc_counter(data_completion_cntr);
    }

    /** @brief Initiate matching between two communications.
     * @details This class calls out to ::Match::Protocol::sender_eager(). Does not use a
     * Phase B communicator.
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_eager(data_entry.get_rma_iov_addr(), base_req, comm_a);
    }

protected:
    /** @brief Queues up two deferred work queue entries, using the global triggering
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
        data_entry.set_threshold(trigger_cntr.get_next_value());
        // Adjust the triggering counter to use
        data_entry.set_trigger_counter(trigger_cntr);

        // Queue up send of user data
        // data_entry.print();
        libfab.queue_work(data_entry.get_dwqe());

        // Queue up completion DWQ
        queue_local_completion();

        return TriggerStatus::GLOBAL_BUMP;
    }

    /** @brief Deferred work queue entry for the RMA operation that moves the user
     * buffer. */
    RMAEntry data_entry;

    /** @brief Allocated completion counter (for the RMA operation) */
    struct fid_cntr* data_completion_cntr;
};

/** @brief Implementation of Communication::Protocol::RNDV send operations.
 * @details Aligns with Communication::Operation::SEND operations. This method works by
 * creating two deferred work queue entries. The first, triggered by the GPU AND a signal
 * of readiness from the peer receive, is for moving the data to the other side. The
 * second, chained off the completion of the first, performs a local atomic increment on
 * the completion buffer so the GPU can watch for when the operation is done.
 *
 * Is NOT able to be triggered off the global counter.
 */
class CXISend : public CXIRSend
{
public:
    /** @brief Constructor for the regular send operations (using a rendezvous protocol).
     * @details This class lets the parents (::CXIRSend and ::CXILocalCompletion) set up
     * most of the deferred work queue objects. Once that is done, it creates a
     * ::CXICounter triggering counter and assigns it to the first deferred work queue
     * entry. It then allocates pinned-host memory for the "clear-to-send" buffer, which
     * will track the "ready" from the remote process via a libfabric counter set up to
     * track FI_REMOTE_WRITE actions. The peer is given the memory region key during the
     * exchange in the matching phase.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     */
    CXISend(Communication::Request& user_request, CompletionBufferFactory& buffers,
            LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRSend(user_request, buffers, _libfab, self),
          send_trigger_cntr(_libfab),
          cts_buffer(Memory::gpu_host_resource(),
                     CompletionBufferFactory::DEFAULT_ITEM_SIZE)
    {
        data_entry.set_trigger_counter(send_trigger_cntr);

        /* Setup CTS buffer */
        cts_mr = _libfab.create_mr_with_counter(
            cts_buffer.get(), CompletionBufferFactory::DEFAULT_ITEM_SIZE, FI_REMOTE_WRITE,
            FI_MR_ALLOCATED | FI_RMA_EVENT, send_trigger_cntr.counter, FI_REMOTE_WRITE);

        /* Describe CTS buffer for the peer */
        cts_details = {0, 1, fi_mr_key(cts_mr)};
    }

    /** @brief Destructor that closes its libfabric memory region.
     * @details The allocated GPU host memory will be freed automatically after the
     * destructor.
     */
    ~CXISend()
    {
        /* Close the MR; cts_buffer is freed afterwards, as a member */
        check_libfabric(fi_close(&(cts_mr)->fid));
    }

    /** @brief Initiate matching between two communications (that want to use the
     * rendezvous protocol).
     * @details This class calls out to ::Match::Protocol::sender_rndv().
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @param [in] comm_b Communication::MPIS_COMM_WORLD duplicate for Phase B
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_rndv(data_entry.get_rma_iov_addr(), &cts_details,
                                     base_req, comm_a, comm_b);
    }

    /** @brief Generate a ::GPUTriggerDescription for the operation.
     * @details Will call ::CXICounter::up_use_count() on ::CXISend::send_trigger_cntr.
     * @return A ::GPUTriggerDescription on how to trigger the start of the operation.
     */
    GPUTriggerDescription get_gpu_trigger() override
    {
        send_trigger_cntr.up_use_count();
        return (GPUTriggerDescription)(send_trigger_cntr.gpu_mmio_addr);
    }

protected:
    /** @brief Queues up two deferred work queue entries to carry out a regular send
     * operation.
     * @details Unlike ::CXIRSend, before this operation can send its data, it needs to
     * know that the other side is ready. To do this, we use the class-specific
     * ::CXICounter that has been set up to track FI_REMOTE_WRITE operations to the
     * clear-to-send buffer (::CXISend::cts_buffer). Once one of those has happened, and
     * the local GPU has triggered the counter to be incremented, the deferred work queue
     * item for the send will be triggered. The local completion operation remains the
     * same as the parent class.
     *
     * The fact that the corresponding receive operation will block its GPU to wait on
     * the completion of the send combined with the semantic reason that it is not allowed
     * to queue two start operations on a request without a wait in between is the reason
     * why this function can get away with "send_trigger_cntr.get_next_value() * 2" as the
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
        data_entry.set_threshold(send_trigger_cntr.get_next_value() * 2);

        // Queue up send of user data
        // data_entry.print();
        libfab.queue_work(data_entry.get_dwqe());

        // Queue up completion DWQ
        queue_local_completion();

        return TriggerStatus::EXTRA_BUMP;
    }

private:
    /** @brief Separate counter for triggering send operations. */
    CXICounter send_trigger_cntr;
    /** @brief Buffer wrapper for the clear-to-send pinned-host buffer. */
    Memory::OwnedBuffer cts_buffer;
    /** @brief Libfabric memory region handle for the clear-to-send buffer. */
    fid_mr* cts_mr;
    /** @brief Details about the "clear-to-send" buffer for this send operation. */
    struct fi_rma_ioc cts_details;
};

/** @brief Implementation of Communication::Protocol::IPC send operations.
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
    /** @brief Constructor for on-node eager/rsend operations.
     * @details Initializes ::CXIRSendShared::handles_open to true.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     */
    CXIRSendShared(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers),
          peer_buffer_ptr(nullptr),
          peer_completion_ptr(nullptr),
          handles_open(true)
    {
    }

    /** @brief Destructor. Closes the IPC handles if they were opened. */
    ~CXIRSendShared()
    {
        if (!handles_open)
        {
            check_gpu(hipIpcCloseMemHandle(peer_buffer_ptr));
            check_gpu(hipIpcCloseMemHandle(peer_completion_ptr));
        }
    }

    /** @brief Initiate matching between two on-node, ready-based communications.
     * @details This class calls out to ::Match::Protocol::sender_hip_ipc(). Only uses
     * phase A communication during matching.
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::sender_hip_ipc(ipc_data, base_req, comm_a);
    }

protected:
    /** @brief Carry out enqueueing the actions needed to perform an IPC-based send
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
     * before the data is moved. This is fixed in ROCm 7.14+ (potentially all ROCm 7
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
        if (handles_open)
        {
            base_req.out("Opening handles for send request!");
            force_gpu(hipIpcOpenMemHandle(&peer_buffer_ptr, ipc_data[0].handle,
                                          hipIpcMemLazyEnablePeerAccess));
            force_gpu(hipIpcOpenMemHandle(&peer_completion_ptr, ipc_data[1].handle,
                                          hipIpcMemLazyEnablePeerAccess));
            handles_open = false;
        }
        Print::out("<E> Using Offsets for IPC:", ipc_data[0].offset, ipc_data[1].offset,
                   base_req.get_size_of_buffer(), peer_buffer_ptr);

        void* peer_true_buffer = (char*)peer_buffer_ptr + ipc_data[0].offset;
        force_gpu(hipMemcpyDtoDAsync(peer_true_buffer, base_req.send_buffer,
                                     base_req.get_size_of_buffer(), *the_stream));

        // Assuming that this stages "num_times_started", otherwise race condition!
        void* peer_true_completion = (char*)peer_completion_ptr + ipc_data[1].offset;
        force_gpu(hipMemcpyHtoDAsync(peer_true_completion, &num_times_started,
                                     sizeof(num_times_started), *the_stream));

        return TriggerStatus::DONE;
    }

private:
    /** @brief GPU memory handle to remote peer's data buffer. */
    void* peer_buffer_ptr;
    /** @brief GPU memory handle to remote peer's completion buffer. */
    void* peer_completion_ptr;
    /** @brief Have the IPC handles been opened or not. */
    bool handles_open;
    /** @brief Details about the peer's buffers (as IPC handles)
     * @details Index 0 is the remote buffer and index 1 is the remote completion buffer.
     */
    std::array<Match::Protocol::IPCBundle, 2> ipc_data;
};

/** @brief Implementation of Communication::Protocol::SELF for send operations.
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
    /** @brief Constructor for eager/rsend operations to the same process.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     */
    CXIRSendSelf(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
    }

    /** @brief Initiate matching between two on-same-process, ready-based communications.
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
    /** @brief Carry out enqueueing the actions needed to perform a self-send operation.
     * @details The implementation of this is two asynchronous HIP mem-copies to the
     * addresses obtained in the matching phase. Same reasoning for the return code as
     * ::CXIRSendShared::start_derived().
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

/** @brief Implementation of Communication::Protocol::CREDIT send operations.
 * @details This protocol attempts to combine eager and rendezvous protocols to get the
 * best of both. By pre-allocating some buffer space ahead of time, this protocol can
 * ensure that it can send a few messages in an eager manner. When these run out, it falls
 * back to needing "clear-to-send" from the peer process. The number of these credit slots
 * is controlled by ::Communication::MAX_CREDIT_SLACK. On the sending side, this value is
 * used to allocate the number of "clear-to-send" buffers. On the receiving side only a
 * single buffer (of size ::Communication::MAX_CREDIT_SLACK * user buffer size) and
 * libfabric memory key are created.
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
    /** @brief Constructor for credit-based send operations. Allocates credit
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
            CXICounter          temp_credit_counter(_libfab);
            Memory::OwnedBuffer temp_credit_buffer(
                Memory::gpu_host_resource(), CompletionBufferFactory::DEFAULT_ITEM_SIZE);
            fid_mr* temp_credit_mr = _libfab.create_mr_with_counter(
                temp_credit_buffer.get(), CompletionBufferFactory::DEFAULT_ITEM_SIZE,
                FI_REMOTE_WRITE, FI_MR_ALLOCATED | FI_RMA_EVENT,
                temp_credit_counter.counter, FI_REMOTE_WRITE);
            Print::out("CREDIT MR SLOT:", temp_credit_buffer.get(),
                       temp_credit_counter.gpu_mmio_addr, temp_credit_mr);
            credit_slots.emplace_back(std::move(temp_credit_counter),
                                      std::move(temp_credit_buffer), temp_credit_mr);
            local_credit_iocs[index] = {0, 1, fi_mr_key(temp_credit_mr)};
        }
    }

    /** @brief Destructor that closes the memory region for each credit slot.
     * @details The buffers for the credit slots will be freed after the class destructor
     * wraps up.
     */
    ~CXISendCredit()
    {
        /* Close the MRs; the buffers are freed afterwards, with credit_slots */
        for (auto& slot : credit_slots)
        {
            check_libfabric(fi_close(&(std::get<2>(slot))->fid));
        }
    }

    /** @brief Initiate matching between two credit-based communications.
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
        Match::Protocol::sender_credit(&peer_staging_iov, local_credit_iocs, base_req,
                                       comm_a, comm_b);
    }

    /** @brief Generate a ::GPUTriggerDescription for the operation.
     * @details Will call ::CXICounter::up_use_count() on one of the ::CXICounter objects
     * in ::CXISendCredit::credit_slots, according to which credit slot is in use. The
     * chosen counter is also the one whose MMIO address is returned to be triggered. Note
     * that the chosen credit slot depends on ::CXIRequest::num_times_started.
     * @return A ::GPUTriggerDescription on how to trigger the start of the operation.
     */
    GPUTriggerDescription get_gpu_trigger() override
    {
        size_t slot_index = (num_times_started - 1) % Communication::MAX_CREDIT_SLACK;
        auto&  triggered  = std::get<0>(credit_slots.at(slot_index));
        triggered.up_use_count();
        return (GPUTriggerDescription)(triggered.gpu_mmio_addr);
    }

protected:
    /** @brief Carry out enqueueing the actions needed to perform a credit-based send
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
        size_t slot_index = (num_times_started - 1) % Communication::MAX_CREDIT_SLACK;
        size_t threshold =
            ((num_times_started - 1) / Communication::MAX_CREDIT_SLACK) * 2 + 1;

        /* Update work entry to the correct remote buffer data */
        auto     user_buffer_size = base_req.get_size_of_buffer();
        uint64_t offset           = slot_index * user_buffer_size;
        peer_staging_iov.addr     = offset;
        peer_staging_iov.len      = user_buffer_size;
        Print::out("Using:", peer_staging_iov.addr, peer_staging_iov.len,
                   peer_staging_iov.key);
        data_entry.set_rma_iov(peer_staging_iov);
        /* Update work entry to use current credit threshold */
        data_entry.set_threshold(threshold);
        /* Adjust the triggering counter to use the credit counter */
        data_entry.set_trigger_counter(std::get<0>(credit_slots.at(slot_index)));

        // Queue up send of user data
        // data_entry.print();
        libfab.queue_work(data_entry.get_dwqe());

        // Queue up completion DWQ
        queue_local_completion();

        return TriggerStatus::EXTRA_BUMP;
    }

private:
    /** @brief Details about the local credit "clear-to-send" buffers.
     * @details Each tuple holds the following values: Index 0 is the ::CXICounter
     * attached to the buffer; Index 1 is a Memory::OwnedBuffer wrapper of the raw pointer
     * for the buffer; Index 2 is the libfabric memory region of the buffer. Vector is of
     * size ::Communication::MAX_CREDIT_SLACK.
     */
    std::vector<std::tuple<CXICounter, Memory::OwnedBuffer, fid_mr*>> credit_slots;

    /** @brief Details about the remote credit buffer.
     * @details Offset is adjusted per enqueued start operation.
     */
    struct fi_rma_iov peer_staging_iov;

    /** @brief A set of fi_rma_ioc objects detailing the credit availability buffers on
     * the send side.
     * @details Only used in setup and matching phase.
     * @sa Match::Protocol::CreditBundle
     */
    Match::Protocol::CreditArray local_credit_iocs;
};

/** @brief A common base class for some receive operations.
 * @details This base class sets up a libfabric memory region on a provided buffer (could
 * be the user's buffer, see ::CXIRecvEager, or on some sort of control message buffer,
 * see ::CXIRecvCredit), as well as a libfabric counter to track FI_REMOTE_WRITE
 * operations to that buffer. This class also allocates a libfabric counter (no progress
 * tracking), and sets up CXILocalCompletion::local_completion_entry to be triggered on
 * this counter (CXIRecvBase::remote_write_cntr). Mostly a helper "parent" class that
 * reduces duplication of setting up a memory region to track remote writes (that then
 * triggers a dwq entry).
 */
class CXIRecvBase : public CXILocalCompletion
{
public:
    /** @brief Constructor that sets up a memory region to track remote writes to.
     * @details The remainder of this method aligns with the class details. The specific
     * libfabric flags align with what originally worked in testing.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     * @param [in] mr_buffer Address to use for the buffer that will be registered and
     * should track remote writes on.
     * @param [in] mr_size The size of the buffer to be registered.
     */
    CXIRecvBase(Communication::Request& user_request, CompletionBufferFactory& buffers,
                LibfabricInstance& _libfab, fi_addr_t self, void* mr_buffer,
                size_t mr_size)
        : CXILocalCompletion(user_request, buffers, _libfab, self),
          remote_write_cntr(_libfab.alloc_counter(false))  // registered with MR
    {
        recv_mr = _libfab.create_mr_with_counter(mr_buffer, mr_size, FI_REMOTE_WRITE,
                                                 FI_MR_ALLOCATED | FI_RMA_EVENT,
                                                 remote_write_cntr, FI_REMOTE_WRITE);

        mr_rma_iov = {0, mr_size, fi_mr_key(recv_mr)};

        local_completion_entry.set_trigger_counter(remote_write_cntr);
    }

    /** @brief Destructor that closes its memory region and associated counter. */
    ~CXIRecvBase()
    {
        // Free MR (before the counter bound to it)
        force_libfabric(fi_close(&(recv_mr)->fid));
        libfab.dealloc_counter(remote_write_cntr);
    }

protected:
    /** @brief Counter that is setup to trigger the local completion deferred
     * work queue action upon a remote write action happening. */
    struct fid_cntr* remote_write_cntr;
    /** @brief Memory region that will watch for remote writes to. */
    struct fid_mr* recv_mr;
    /** @brief Details about the memory region (offset, size, key) and given to peers that
     * will be writing to it. */
    struct fi_rma_iov mr_rma_iov;
};

/** @brief Implementation of Communication::Protocol::EAGER receive operations.
 * @details Usually aligns with peer Communication::Operation::RSEND operations. This
 * receive operation does not make any additional deferred work queue items (but does use
 * the one created by CXILocalCompletion).
 *
 * This operation is not triggered by the global counter (there is no action to trigger).
 * Instead, local completion notification is triggered by the receive physically
 * happening, which increments a counter that triggers the local completion operation.
 */
class CXIRecvEager : public CXIRecvBase
{
public:
    /** @brief Constructor for eager receive operations.
     * @details Gives the user's buffer to the parent CXIRecvBase constructor to track
     * remote writes to. Otherwise just passes the other variables to the parent
     * constructor.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     */
    CXIRecvEager(Communication::Request& user_request, CompletionBufferFactory& buffers,
                 LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRecvBase(user_request, buffers, _libfab, self, user_request.recv_buffer,
                      user_request.get_size_of_buffer())
    {
    }

    /** @brief Initiate matching between two communications.
     * @details This class calls out to ::Match::Protocol::receiver_eager(). Only uses
     * phase A communication during matching.
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::receiver_eager(&mr_rma_iov, base_req, comm_a);
    }

protected:
    /** @brief Sets up GPU actions for the eager receive operation.
     * @details Calls CXILocalCompletion::queue_local_completion() and then returns. Does
     * not use any of the function parameters.
     * @return ::TriggerStatus::NOT_NEEDED to indicate that this operation does not need
     * any triggering to start its operation.
     */
    TriggerStatus start_derived(CXICounter&, hipStream_t*) override
    {
        queue_local_completion();
        return TriggerStatus::NOT_NEEDED;
    }
};

/** @brief Implementation of Communication::Protocol::RNDV receive operations.
 * @details Using CXIRecvEager as its base, this class adds in a "clear to send" control
 * message into its workflow. This message requires a deferred work queue to be carried
 * out, and the details about the send-side's CTS buffer will need to be obtained during
 * matching. The completion of the CTS deferred work queue is not currently used for
 * anything (though its counter is marked to track progress).
 *
 * The CTS operation is able to be triggered off the global counter.
 */
class CXIRecvRndv : public CXIRecvEager
{
public:
    /** @brief Constructor for rendezvous receive operations.
     * @details Aside from calling the parent ::CXIRecvEager constructor, this constructor
     * sets up the deferred work queue item (minus the triggering counter and threshold)
     * for the "clear-to-send" operation. To assist in this, a counter (with progress
     * tracking) is created and set as the completion counter. This counter is otherwise
     * unused.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     */
    CXIRecvRndv(Communication::Request& user_request, CompletionBufferFactory& buffers,
                LibfabricInstance& _libfab, fi_addr_t self)
        : CXIRecvEager(user_request, buffers, _libfab, self),
          cts_completion_cntr(_libfab.alloc_counter(true)),
          cts_entry(_libfab.ep, _libfab.get_peer(user_request.cw_peer))
    {
        cts_entry.set_completion_counter(cts_completion_cntr);
    }

    /** @brief Deallocate the libfabric counter created. */
    ~CXIRecvRndv()
    {
        libfab.dealloc_counter(cts_completion_cntr);
    }

    /** @brief Initiate matching between two communications.
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
        Match::Protocol::receiver_rndv(&mr_rma_iov, cts_entry.get_rma_ioc_addr(),
                                       base_req, comm_a, comm_b);
    }

protected:
    /** @brief Queues up the deferred work queue items for the CTS and local completion.
     * @details First the next threshold for the counter is set, then the triggering
     * counter is set in the CTS deferred work queue, and then its deferred work queue
     * entry is queued up. Local completion is handled by a call to
     * CXILocalCompletion::queue_local_completion().
     * @param [in] trigger_cntr The global hardware counter that will be used to trigger
     * communication.
     * @param [in] the_stream The GPU stream that should be enqueued to. Unused.
     * @return ::TriggerStatus::GLOBAL_BUMP (for the CTS action to be triggered).
     * @sa CXICounter::get_next_value()
     */
    TriggerStatus start_derived(CXICounter&  trigger_cntr,
                                hipStream_t* the_stream) override
    {
        Print::out("Queue CTS to Libfabric!");
        // Update threshold of chained things
        cts_entry.set_threshold(trigger_cntr.get_next_value());
        // Adjust the triggering counter to use
        cts_entry.set_trigger_counter(trigger_cntr);
        libfab.queue_work(cts_entry.get_dwqe());

        queue_local_completion();
        return TriggerStatus::GLOBAL_BUMP;
    }

private:
    /** @brief Completion counter for "clear-to-send" action. Only used to track progress
     * on DWQ entry. */
    struct fid_cntr* cts_completion_cntr;
    /** @brief Deferred work queue entry for the atomic operation that moves
     * "clear-to-send" signal to the peer. */
    AtomicEntry cts_entry;
};

/** @brief Implementation of Communication::Protocol::IPC for receive operations (matched
 * with on-node ready send operations).
 * @details This class's main workload is in the initialization and matching phases. Once
 * this class has set up HIP IPC and given it to the sending side, this class does nothing
 * else. Does not use any libfabric items, including counters and deferred work queue
 * entries. However, this class still does allocate a completion buffer, but completions
 * are generated through hipMemcopy actions on the sending side.
 *
 * Note that this class is only used for ready sends. Regular sends do not match with this
 * receive class.
 *
 * This class does not need to use the global counter for triggering.
 */
class CXIRecvShared : public CXIRequest
{
public:
    /** @brief Constructor for on-node receive operations.
     * @details This method registers the user's buffer to get its IPC handles and obtains
     * the IPC handles for the completion buffer. Details about both of these handles are
     * prepared to be shared with the sender side. Note that because
     * ::CompletionBufferFactory allocates all buffers in one allocation, registration of
     * the completion buffer is not done here. Instead, this method must figure out the
     * offset to give to the peer.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @sa ::CompletionBufferFactory::get_ipc_handle()
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

    /** @brief Initiate matching between two communications.
     * @details This class calls out to ::Match::Protocol::receiver_hip_ipc(). Only uses
     * phase A communication during matching.
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::receiver_hip_ipc(ipc_data, base_req, comm_a);
    }

protected:
    /** @brief No-op for this type of receive operation. (All sender driven)
     * @details None of the provided parameters are used.
     * @return ::TriggerStatus::NOT_NEEDED since no GPU actions are needed for this
     * protocol.
     */
    TriggerStatus start_derived(CXICounter&, hipStream_t*) override
    {
        return TriggerStatus::NOT_NEEDED;
    }

private:
    /** @brief Helper array for organizing the data to be sent during matching.
     * @details The data in this array is as follows: Index 0 holds details about the
     * user's buffer and index 1 holds details about the completion buffer.
     * @sa Match::Protocol::IPCBundle
     */
    std::array<Match::Protocol::IPCBundle, 2> ipc_data;
};

/** @brief Implementation of Communication::Protocol::SELF for receive operations (matched
 * with on-node ready send operations to the same process).
 * @details This class works similarly to ::CXIRecvShared except that it does not need to
 * use HIP IPC buffer handles. Instead, since both sides are on the same process, they can
 * simply exchange raw pointers. Similarly, this class also does all of its work in
 * initialization and matching, and does not add anything to the GPU (aside from the basic
 * waiting kernel).
 * Data movement is carried out only by the sender.
 */
class CXIRecvSelf : public CXIRequest
{
public:
    /** @brief Constructor for on-node, same-process receive operations.
     * @details Fills out CXIRecvSelf::self_data with details about the user buffer's
     * address and its completion buffer address.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     */
    CXIRecvSelf(Communication::Request& user_request, CompletionBufferFactory& buffers)
        : CXIRequest(user_request, buffers)
    {
        self_data[0] = user_request.recv_buffer;
        self_data[1] = completion_buffer.address;
        Print::out("Self wants to go to:", self_data[0], self_data[1]);
    }

    /** @brief Initiate matching between two communications.
     * @details This class calls out to ::Match::Protocol::receiver_self().
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @param [in] comm_b Communication::MPIS_COMM_WORLD duplicate for Phase B
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        Match::Protocol::receiver_self(self_data, base_req, comm_a, comm_b);
    }

protected:
    /** @brief No-op for this type of receive operation. (All sender driven)
     * @details None of the provided parameters are used.
     * @return ::TriggerStatus::NOT_NEEDED since no GPU actions are needed for this
     * protocol.
     */
    TriggerStatus start_derived(CXICounter&, hipStream_t*) override
    {
        return TriggerStatus::NOT_NEEDED;
    }

private:
    /** @brief Helper array for organizing the data to be sent during matching.
     * @details The data in this array is as follows: Index 0 holds details about the
     * user's buffer and index 1 holds details about the completion buffer.
     * @sa Match::Protocol::SelfBundle
     */
    Match::Protocol::SelfBundle self_data;
};

/** @brief Implementation of Communication::Protocol::CREDIT for receive operations
 * @details In terms of libfabric deferred work queue entries, this class behaves
 * similarly to ::CXIRecvRndv. That is, there is one deferred work queue for the CTS and
 * one for the local completion. However, unlike that class, this class makes its own
 * triggering counter, and has to cycle where to the send the CTS to. Further, the CTS
 * are sent AFTER the receive is processed, since these "clear-to-send" act more like
 * "credit-ready". Each time the receive operation is started, it cycles which remote CTS
 * buffer it will write to.
 * The biggest difference for this class is that it provides its own wait kernel, and does
 * NOT use the normal waiting kernel used in CXIQueue::enqueue_waitall().
 * Credit based operations are used for off-node regular sends that are at or below the
 * rendezvous protocol cutoff (::Communication::CREDIT_SIZE_CUTOFF).
 *
 * This class private inherits from Memory::OwnedBuffer so that it can safely create the
 * before the ::CXIRecvBase parent class attempts to register the buffer.
 *
 * @sa CXISendCredit
 */
class CXIRecvCredit : private Memory::OwnedBuffer, public CXIRecvBase
{
public:
    /** @brief Constructor for credit-based receive operations.
     * @details First, this constructor allocates GPU device memory to use for the "eager"
     * style buffers this method uses. All buffers (of count
     * ::Communication::MAX_CREDIT_SLACK) are lumped together into a single allocation.
     * This big buffer is then given to the parent ::CXIRecvBase so that it can be
     * assigned a counter to track remote writes on. Next, a local triggering counter is
     * created, and normal deferred work queue entry setup is done for the CTS. The CTS is
     * triggered by the local triggering counter, and its completion counter is registered
     * to track progress on.
     * @param [in] user_request Details about the operation to carry out.
     * @param [in, out] buffers The factory for allocating completion buffers from.
     * @param [in, out] _libfab Libfabric container to enqueue the deferred work
     * operations to.
     * @param [in] self Address of this process to use for local completion atomic
     * increment.
     */
    CXIRecvCredit(Communication::Request& user_request, CompletionBufferFactory& buffers,
                  LibfabricInstance& _libfab, fi_addr_t self)
        : Memory::OwnedBuffer(Memory::gpu_device_resource(), credit_size(user_request)),
          CXIRecvBase(user_request, buffers, _libfab, self, Memory::OwnedBuffer::get(),
                      credit_size(user_request)),
          credit_trigger_cntr(_libfab),
          credit_completion_cntr(_libfab.alloc_counter(true)),
          credit_entry(_libfab.ep, _libfab.get_peer(user_request.cw_peer))
    {
        // Adjust the triggering counter to use
        credit_entry.set_trigger_counter(credit_trigger_cntr);
        credit_entry.set_completion_counter(credit_completion_cntr);
    }

    /** @brief Deallocates the created libfabric counter. */
    ~CXIRecvCredit()
    {
        libfab.dealloc_counter(credit_completion_cntr);
    }

    /** @brief Initiate matching between two communications.
     * @details This class calls out to ::Match::Protocol::receiver_credit().
     * @param [in] comm_a Communication::MPIS_COMM_WORLD duplicate for Phase A
     * communication.
     * @param [in] comm_b Communication::MPIS_COMM_WORLD duplicate for Phase B
     * communication.
     * @sa ::Match
     */
    void match(MPI_Comm comm_a, MPI_Comm comm_b) override
    {
        /* Start requests to exchange from peer */
        Match::Protocol::receiver_credit(&mr_rma_iov, peer_credit_iocs, base_req, comm_a,
                                         comm_b);
    }

    /** @brief Perform GPU actions related to waiting on the operation.
     * @details This method first figures out which "credit" buffer this receive operation
     * will need to copy out from. It then enqueues the custom kernel that waits for the
     * data to come in, copies it over to the receive buffer, and then triggers the CTS
     * back to the sender.
     * @param [in] the_stream The GPU stream that should be enqueued to.
     * @return ::WaitStatus::NOT_NEEDED as this method created its own kernel to do the
     * waiting in.
     * @sa ::credit_recv()
     */
    WaitStatus wait_gpu(hipStream_t* the_stream) override;

protected:
    /**
     * @brief Prepares the deferred work queue items for the receiving side of the credit
     * protocol.
     * @details This method does not use any of the provided parameters (uses its own
     * counter, and does not add anything to the stream here).
     * The remote "credit" CTS buffer is selected here, and put into the deferred work
     * queue item for that control message. Likewise, the
     * CXILocalCompletion::queue_local_completion() is called to prepare the deferred work
     * queue item that tracks local completion.
     * @return ::TriggerStatus::NOT_NEEDED to indicate that this operation does not need
     * any triggering assistance. This protocol will trigger the CTS on its own
     * at the end of its waiting kernel.
     */
    TriggerStatus start_derived(CXICounter&, hipStream_t*) override
    {
        Print::out("Queue Credit to Libfabric!");
        // Update which credit slot we want to write to
        credit_entry.set_rma_ioc(peer_credit_iocs.at(
            ((num_times_started - 1) % Communication::MAX_CREDIT_SLACK)));
        // Update threshold of chained things
        credit_entry.set_threshold(credit_trigger_cntr.get_next_value());
        // Queue up remote credit DWQ
        libfab.queue_work(credit_entry.get_dwqe());

        // Queue up local completion DWQ
        queue_local_completion();

        return TriggerStatus::NOT_NEEDED;
    }

private:
    /** @brief Helper method for calculating the size of the staging credit buffers.
     * @param [in] user_request The Communication::Request that holds the size of the
     * user's buffer.
     * @return In bytes, the size of the user's buffer multiplied by
     * Communication::MAX_CREDIT_SLACK
     */
    static size_t credit_size(Communication::Request& user_request)
    {
        return user_request.get_size_of_buffer() * Communication::MAX_CREDIT_SLACK;
    }

    /** @brief Helper method to get the underlying pointer to the staging credit buffers.
     * @return The pointer to the staging credit buffers.
     */
    void* staging_buffers() const
    {
        return Memory::OwnedBuffer::get();
    }

    /** @brief Custom triggering counter for the "credit" CTS. */
    CXICounter credit_trigger_cntr;
    /** @brief Details about the sender's "credit" CTS buffers. */
    Match::Protocol::CreditArray peer_credit_iocs;
    /** @brief Completion counter for "clear-to-send" action. Only used to track progress
     * on DWQ entry. */
    struct fid_cntr* credit_completion_cntr;
    /** @brief Deferred work queue entry for the atomic operation that moves
     * "clear-to-send" signal to the peer. */
    AtomicEntry credit_entry;
};

#endif
