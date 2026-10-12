#ifndef ST_CXI_QUEUE
#define ST_CXI_QUEUE

#include <map>
#include <numeric>
#include <vector>

#include "misc/print.hpp"
#include "queues/HIPQueue.hpp"
#include "queues/cxi/libfabric_wrappers.hpp"
#include "queues/cxi/requests/request_base.hpp"
#include "safety/gpu.hpp"
#include "safety/mpi.hpp"

/**
 * @file CXIQueue.hpp
 * @brief Code for the implementation of a CXI Libfabric based Queue backend.
 */

/**
 * @brief A backend built around the CXI libfabric provider and its extensions that enable
 * GPU stream triggered communication.
 * @details This class uses the HIPQueue backend to inherit basic operation handling for
 * collective Communication::Operation. This class has no interaction with its
 * parent class aside from sharing the underlying stream object and the collective
 * HIPQueue path.
 *
 * The rest of the class makes heavy use of the CXI libfabric provider to accomplish the
 * stream-triggering code. Members of this class are documented as appropriate, but see
 * the CLUSTER 2026 paper for a more collected description of everything.
 */
class CXIQueue : public HIPQueue
{
public:
    /** @brief Alias type for an owning pointer to a ::CXIRequest. */
    using CXIObjects = std::unique_ptr<CXIRequest>;

    /**
     * @brief Constructor for CXI-based ::Queue implementation.
     * @details This constructor is a collective operation on all of the processes in
     * CXIQueue::comm_base (which is currently all processes in the
     * Communication::MPIS_COMM_WORLD) for two reasons. The first is that several
     * communicator duplications happen, and the second is that all processed need to
     * participate in an allgather to get every rank's CXI address for libfabric
     * connection building.
     *
     * Documentation on how and what libfabric objects are setup is left to
     * LibfabricInstance::initialize().
     *
     * After setup, this object also create a shared CXICounter for triggering multiple
     * operations at once and registers the entire set of completion buffers with
     * libfabric.
     *
     * @param [in] stream_addr A pointer to a HIP GPU stream that will be used to enqueue
     * start and wait kernels onto.
     */
    CXIQueue(hipStream_t* stream_addr) : HIPQueue(stream_addr), comm_base(MPIS_COMM_WORLD)
    {
        Print::out("CXI Queue init-ed");
        force_mpi(MPI_Comm_rank(comm_base, &my_rank));
        Print::out("Starting MPI Comm Dupes");
        force_mpi(MPI_Comm_dup(comm_base, &protocol_phase));
        force_mpi(MPI_Comm_dup(comm_base, &match_phase_a));
        force_mpi(MPI_Comm_dup(comm_base, &match_phase_b));
        Print::out("Starting Allgather to get CXI address data");
        libfab.initialize(match_phase_a);
        Print::out("Creating on-node communicator");
        force_mpi(MPI_Comm_split_type(comm_base, MPI_COMM_TYPE_SHARED, my_rank,
                                      MPI_INFO_NULL, &on_node_peers));

        // Global counter for when requests can share a triggering counter
        the_gpu_counter = std::make_unique<CXICounter>(libfab);

        // Register MR
        my_buffer.register_mr(libfab);
    }

    /** @brief Destructor that frees up all internal objects.
     * @details The call also makes a collective call to MPI_Barrier on
     * CXIQueue::comm_base (which is currently Communication::MPIS_COMM_WORLD) just in
     * case there are odd errors that arise from one side freeing network resources while
     * other sides are still using them.
     *
     * Aside from freeing all MPI communicators made during the constructor, the libfabric
     * objects made and managed by this class are also freed at this time. The specific
     * order in which things are cleared or reset is to align with the order in which
     * libfabric objects need to be freed up (aside from CXIQueue::libfab, which needs to
     * be freed last automatically).
     */
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

    /**
     * @brief Enqueue a single operation's start onto the stream.
     * @details Currently implemented as single call to CXIQueue::enqueue_startall.
     * @param [in] request The Communication::Request to start.
     */
    void enqueue_operation(std::shared_ptr<Request> request) override
    {
        enqueue_startall({request});
    }

    /**
     * @brief Enqueue a start operation for set of communication operations.
     * @details This function is how stream-triggering kernels are put onto a GPU stream.
     * Before any kernels are added, if the system is an MI250X system (currently decided
     * by if USE_TIOGA is defined) and any operation needs the GPU buffers to be flushed
     * (i.e., so the NIC sends the correct data, see
     * Communication::Request::needs_gpu_flush()), an additional kernel
     * (CXIQueue::flush_memory()) is added.
     *
     * All collective operations are deferred to HIPQueue::enqueue_operation(), and are
     * not added to CXIQueue::active_requests.
     *
     * All point-to-point operations in @p requests are then checked for their desired
     * triggering mechanism (::TriggerStatus) after calling CXIRequest::start_cpu(). If
     * the operation is unable to use the global triggering counter passed to that method,
     * the code collects all of the counters to be triggered. The global counter will only
     * be added the first time it is requested. CXIRequest that need no counters (but
     * still want to be waited on; ::TriggerStatus::NOT_NEEDED) will not add anything to
     * the kernel.
     *
     * If too many counters need to be triggered (see
     * CompletionBufferFactory::MAX_GPU_COMPLETIONS and ::add_to_counters()), then
     * multiple kernels will be launched.
     *
     * At the end, the gpu counter's (CXIQueue::the_gpu_counter)  next "threshold" is set
     * (by CXICounter::up_use_count(), if it was used) and any counter triggers are queue
     * up. Note that if all requests return ::TriggerStatus::DONE, this function may add
     * no additional kernels to the stream (unless a memory flush is needed), and they
     * will not be added to CXIQueue::active_requests (meaning they won't be waited on by
     * the GPU).
     *
     * @param [in] requests The set of requests to enqueue a start for.
     * @sa ::GPUTriggers
     * @sa ::add_to_counters
     */
    void enqueue_startall(std::vector<std::shared_ptr<Request>> requests) override;

    /** @brief Enqueue kernel(s) to wait on the completion of operations.
     * @details This function, like CXIQueue::enqueue_startall, adds kernels related to
     * communication to the GPU stream. For all of the operations that have had their
     * starts enqueued, this function collects all of the completion buffers (and their
     * goal values) to build a ::GPUCompletions object. Like when starts are enqueued, if
     * too many (greater than CompletionBufferFactory::MAX_GPU_COMPLETIONS) are present,
     * multiple kernels will be launched.
     *
     * All active operations will have CXIRequest::wait_gpu() called on them. The ones
     * that return ::WaitStatus::NOT_NEEDED are presumed to have launched their own wait
     * kernel. ::WaitStatus::GLOBAL_KERNEL means the operation's completion buffer will be
     * added into the kernel to be waited on.
     *
     * After launching the kernel(s), CXIQueue::active_requests is cleared.
     *
     * Any collective operation will be deferred to HIPQueue::enqueue_waitall() after the
     * point-to-point operations have been handled.
     *
     * Currently, if USE_TIOGA is set, CXIQueue::flush_memory() is called.
     *
     * @sa ::GPUCompletions
     * @sa ::wait_on_completions()
     */
    void enqueue_waitall() override;

    /** @brief Block the host until synchronized with the GPU.
     * @details Since this ::Queue could have outstanding collective operations, it first
     * calls ::Queue::host_wait() to ensure those are done as well.
     */
    void host_wait() override
    {
        Queue::host_wait();
        Print::out("Waiting on device!");
        force_gpu(hipStreamSynchronize(*my_stream));
    }

    /**
     * @brief Begin the process of matching for a set of communication operations.
     * @details For each of the Communication::Request objects provided, their matching
     * phase will be initiated. If it is a collective operation, it will be handed over to
     * Queue::initiate_match(). Otherwise it will be passed to
     * CXIQueue::exchange_protocol() to setup the MPI communications to exchange
     * protocols.
     *
     * Note that for the CXIQueue, the matching phase for point-to-point operations starts
     * with an exchange of protocols. The call to "match" function (CXIRequest::match())
     * will happen later once those are completed in CXIQueue::finalize_match()
     *
     * @param [in, out] requests The set of Communication::Request objects that will have
     * their matching phase started.
     */
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

    /**
     * @brief Finalize the match between communication operations.
     * @details For point-to-point operations, the matching part of the match phase still
     * needs to occur, so those Communication::Request objects will be handed over to
     * CXIQueue::prepare_cxi_mr_key() where the exact protocol and ::CXIRequest class will
     * be decided for them. Then, depending on which class is decided, the corresponding
     * Match:: functions will be started (via CXIRequest::match()).
     *
     * At the end, the super Queue::finalize_match() will be called to wait on all
     * outstanding MPI communications from the matching phase (for both point-to-point and
     * collective operations).
     *
     * @param [in, out] requests A vector of shared pointers of the Communication::Request
     * that should have their match process finalized.
     */
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

    /**
     * @brief Free the request from any ::CXIQueue resources.
     * @details First removes it from CXIQueue::request_map (if present) before calling
     * Queue::request_free() to remove it from other resources.
     * @param [in, out] request The Communication::Request to be removed.
     */
    void request_free(std::shared_ptr<Request> request) override
    {
        if (request_map.contains(request->getID()))
        {
            request->out("Removing from CXIQueue's cache!");
            request_map.erase(request->getID());
        }
        Queue::request_free(request);
    }

private:
    /**
     * @brief Determine the Communication::Protocol to use for an operation.
     * @details Sets Communication::Request::protocol and calls
     * Communication::Request::start_protocol_exchange(). Performs checks for self
     * communication, communication on a node, and the Communication::Operation involved.
     * Depending on the environment variables set, some protocols may be disabled or have
     * buffer size cutoffs.
     *
     * All Communication::Operation::RECV operations will be given
     * Communication::Protocol::NONE. These operations will have it set once the protocol
     * exchange completes.
     *
     * @param [in, out] req The Communication::Request object to decide the
     * Communication::Protocol for.
     * @sa ::init_env()
     * @sa CXIQueue::protocol_phase
     */
    void exchange_protocol(Request& req);

    /**
     * @brief Continue the matching phase after the protocol has been decided.
     * @details Function name is a hold over from when fewer protocols were supported. Now
     * the function decides which CXIRequest class best fits a Communication::Request
     * based on the decided Communication::Protocol value. Throws if unable to decide a
     * best fit.
     *
     * First blocks on Communication::Request::wait_on_protocol(). Calls
     * CXIRequest::match() to start matching communications before being inserted into
     * CXIQueue::request_map.
     *
     * @param [in, out] req The Communication::Request operation to continue the matching
     * phase for.
     */
    void prepare_cxi_mr_key(Request& req);

    /** @brief Launch a GPU kernel to flush the L2 cache on a GPU.
     * @details Should not be called if not on an MI250X (which can be checked with the
     * define USE_TIOGA).
     */
    void flush_memory();

    /** @brief Libfabric object manager. */
    LibfabricInstance libfab;

    /** @brief The communicator this ::Queue is based off (currently
     * Communication::MPIS_COMM_WORLD) */
    MPI_Comm comm_base;
    /** @brief This process's rank in CXIQueue::comm_base. */
    int my_rank;
    /** @brief Duplicate of CXIQueue::comm_base used for protocol exchanges. */
    MPI_Comm protocol_phase;
    /** @brief Duplicate of CXIQueue::comm_base used for first half of matching
     * communication.
     * @details Currently also used for the allgather used in
     * LibfabricInstance::initialize().
     */
    MPI_Comm match_phase_a;
    /** @brief Duplicate of CXIQueue::comm_base used for second half of matching
     * communication.
     */
    MPI_Comm match_phase_b;
    /** @brief A split of CXIQueue::comm_base on "MPI_COMM_TYPE_SHARED".
     * @details Used for determining if a peer process is on the same node or not.
     */
    MPI_Comm on_node_peers;

    /** @brief Map of Communication::Request::myID to CXIObjects operation
     * @details This map is similar to Queue::request_cache in behavior, but needs a
     * different object type to carry out triggering actions (in current implementation).
     */
    std::map<size_t, CXIObjects> request_map;
    /** @brief A vector of all operations that have had their starts enqueued.
     * @details Currently separate from Queue::entries because this class keeps track of
     * ongoing operations in a different manner. Uses Communication::Request::myID as the
     * key; is a list of all operations that have had their starts enqueued since the last
     * waitall was enqueued.
     */
    std::vector<size_t> active_requests;

    /** @brief Object responsible for allocating completion/ready buffers for all
     * operations. */
    CompletionBufferFactory my_buffer;

    /** @brief Global GPU triggering counter.
     * @details This object represents an optimization that enables multiple communication
     * operations to share a single counter. While this counter has to stay here instead
     * of inside a ::CXIRequest, the optimization does allow for a single triggering
     * kernel to fire many deferred work queue entries.
     */
    std::unique_ptr<CXICounter> the_gpu_counter;
};

#endif
