#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include <string.h>

#include <stdexcept>
#include <vector>

#include "misc/print.hpp"
#include "safety/mpi.hpp"

/**
 * @file request.hpp
 * @brief Header file for the intermediary Communication::Request class.
 */

/** @brief Namespace for abstract concepts of communication. */
namespace Communication
{

/** @brief Abstraction for describing the actual communication of something.
 *  @details Some of the code depends on using Operation::BARRIER as the cutoff point
 * between point to point and collective operations. Anything below this value is
 * considered a point-to-point operation; anything above is considered a collective
 * operation.
 */
enum Operation : int
{
    SEND,      //!< Point-to-point send operation.
    RSEND,     //!< Point-to-point ready-send operation.
    RECV,      //!< Point-to-point receive operation.
    BARRIER,   //!< Collective barrier operation.
    ALLREDUCE  //!< Collective allreduce operation.
};

/** @brief The list of available protocols for Requests to employ.
 * @details The choice of protocol to use is determined by the Queue that is tasked with
 * matching the Request.
 */
enum Protocol : int
{
    NONE = 0,  //!< No protocol, or Request has just been created.
    IPC,       //!< Use IPC-based communication to move data.
    EAGER,     //!< Use an eager approach to moving data.
    CREDIT,    //!< Like eager, but each side has eager "credit" before it needs to wait.
    RNDV,      //!< Rendezvous protocol (wait for other side to be ready before sending).
    SELF,      //!< Protocol for communication where the peer is the calling process.
};

/** @brief How many eager buffers Protocol::CREDIT will use before having to wait for a
 * free buffer. */
constexpr size_t MAX_CREDIT_SLACK = 5;
/** @brief Max message size for using Protocol::CREDIT. */
constexpr size_t CREDIT_SIZE_CUTOFF = 32768;

/** @brief Whether Queue objects may use IPC to move data. Defaults to true.
 * @sa ::init_env()
 */
extern bool IPC_PROTOCOL_ENABLED;
/** @brief Whether Queue object allowed to use credit-based protocols to move data.
 * Defaults to true.
 * @sa ::init_env()
 */
extern bool CREDIT_PROTOCOL_ENABLED;

/** @brief Library duplicate of MPI_COMM_WORLD.
 * @details Created in ::initialize_st(), freed in the MPI_Finalize override of this
 * library.
 */
extern MPI_Comm MPIS_COMM_WORLD;

/** @brief [TODO]
 * @details
 */
extern MPI_Comm MPIS_COMM_PROTOCOL;

/** @brief Enum for recording the details of a GPU buffer.
 * @details The exact type of buffer may influence the GPU actions a Queue will do,
 * depending on the system involved.
 */
enum GPUMemoryType
{
    COARSE = 1,  //!< GPU Buffer is coarse-grained.
    FINE   = 2,  //!< GPU Buffer is fine-grained.
};

/**
 * @brief A container for communication operations.
 * @details The documentation of this class does its best to distinguish between the
 * Request objects of this library and MPI_Request objects, but be sure to use a careful
 * eye when reading the documentation.
 *
 * This class is also essentially a container of parameters given to MPI communication
 * calls to be "replayed" or implemented by a Queue backend. There are also additional
 * helper members that describe more details of the operation that can be used.
 *
 * It is through this object that Queue objects learn about the communication operations
 * created by the C MPIS bindings.
 */
class Request
{
public:
    /**
     * @brief [TODO]
     * @details
     */
    enum class State
    {
        UNMATCHED,  //!< [TODO]
        MATCHING,   //!< [TODO]
        MATCHED,    //!< [TODO]
        ACTIVE,     //!< [TODO]
    };

    State        state = State::UNMATCHED;  //!< [TODO]
    Operation    operation;                 //!< The Operation of the Request.
    Protocol     protocol;     //!< Which Protocol the Request has been assigned.
    void*        send_buffer;  //!< The buffer to use for sends.
    void*        recv_buffer;  //!< The buffer to use for receives.
    MPI_Count    count;        //!< The number of items in the buffer(s).
    MPI_Datatype datatype;     //!< The type of the buffer(s) involved.
    int          peer;         //!< The peer rank (in the original MPI_Comm).
    int          tag;          //!< The tag to use for communication.
    MPI_Comm     comm;         //!< MPI_Comm to use for communication.
    MPI_Info     info;         //!< Info hints provided about operation.
    MPI_Op       op;           //!< MPI_Op to use for reduce operations.
    int          cw_peer;      //!< Peer's rank in ::MPIS_COMM_WORLD.

    /**
     * @brief The general constructor for a Request object.
     * @details This constructor is very broad in the number of things it accepts.
     * Currently only @p op is optional, defaulting to MPI_OP_NULL if not provided.
     * Neither buffer is required and one or both may be nullptr. The info object is
     * checked for the "mpi_memory_alloc_kinds" key. If the value is exactly
     * "rocm:device:fine", then Request::memory_type is set to GPUMemoryType::FINE,
     * otherwise it is set to GPUMemoryType::COARSE (even if the buffer is CPU memory).
     *
     * The peer process is also translated into the rank they are in MPIS_COMM_WORLD
     * (Request::cw_peer) and an ID is assigned to this object as a part of constructing
     * this object (Request::myID).
     *
     * If in a debug build, a printout of the Request object is performed at the end of
     * the constructor.
     *
     * @param [in] _operation The Operation this Request is attempting to perform.
     * @param [in] _send_buffer The buffer address for send-based Operation actions.
     * @param [in] _recv_buffer The buffer address for receive-based Operation actions.
     * @param [in] _count The number of items in the buffer(s) provided.
     * @param [in] _datatype The type of the buffer address(es).
     * @param [in] _peer The MPI rank of the peer process (in @p _comm ).
     * @param [in] _tag The tag to use to match the communication operation.
     * @param [in] _comm The MPI Communicator that this operation happens on.
     * @param [in] _info MPI_Info hints about the communication operation.
     * @param [in] _op MPI_Op to be used in allreduce operations. Optional.
     */
    Request(Operation _operation, void* _send_buffer, void* _recv_buffer,
            MPI_Count _count, MPI_Datatype _datatype, int _peer, int _tag, MPI_Comm _comm,
            MPI_Info _info, MPI_Op _op = MPI_OP_NULL)
        : operation(_operation),
          protocol(Protocol::NONE),
          send_buffer(_send_buffer),
          recv_buffer(_recv_buffer),
          count(_count),
          datatype(_datatype),
          peer(_peer),
          tag(_tag),
          comm(_comm),
          info(_info),
          op(_op),
          cw_peer(resolve_comm_world(_peer)),
          myID(assignID()),
          protocol_request(MPI_REQUEST_NULL)
    {
        constexpr int     string_size = 100;
        char              info_key[]  = "mpi_memory_alloc_kinds";
        std::vector<char> value(string_size, 0);
        int               flag = 0;
        // Pre MPI-4.0
        if (MPI_INFO_NULL != _info)
        {
            force_mpi(
                MPI_Info_get(_info, info_key, string_size - 1, value.data(), &flag));
        }

        if (0 == strcmp(value.data(), "rocm:device:fine"))
        {
            out("Using fine-grained memory!");
            memory_type = GPUMemoryType::FINE;
        }
        else
        {
            out("Using coarse-grained memory!");
            memory_type = GPUMemoryType::COARSE;
        }

        print();
    }

    /** [UPDATE]
     * @brief Return the matched status of a Request object.
     * @return The matched status of a Request object.
     */
    bool is_matched()
    {
        return State::MATCHED <= state;
    }

    /** [UPDATE]
     * @brief Set Request::matched to true. Multiple calls have no effect.
     */
    void set_match()
    {
        state = State::MATCHED;
    }

    /** @brief Access the assigned ID of the Request object.
     * @return The assigned ID of the Request object.
     */
    size_t getID()
    {
        return myID;
    }

    /** @brief Calculate the total size of the buffer involved in the communication.
     * @details Currently, this method returns the size of the final "packed buffer" for
     * non-contiguous datatypes and not the true size of a non-contiguous buffer.
     * Calculates the size of the MPI_Datatype in Request::datatype and value of
     * Request::count.
     * @return The size, in bytes, of the buffer involved in the communication.
     */
    inline size_t get_size_of_buffer()
    {
        if (MPI_DATATYPE_NULL == datatype)
        {
            return 0;
        }
        else
        {
            int size = -1;
            check_mpi(MPI_Type_size(datatype, &size));
            return (size_t)(size * count);
        }
    }

    /** [UPDATE]
     * @brief Prepare the internal vectors for holding a specified number of MPI objects.
     * @details Calling this method multiple times on the same Request object is allowed,
     * but any existing requests are discarded without being freed or completed.
     * @param [in] num The size to make the vectors.
     * @return The underlying pointer of Request::match_requests (for convenient use with
     * the C MPI functions.).
     */
    MPI_Request* get_match_requests(size_t num)
    {
        if (!match_requests.empty())
        {
            throw std::runtime_error(
                "Request already has matching MPI_Requests that were never joined");
        }
        match_requests = std::vector<MPI_Request>(num, MPI_REQUEST_NULL);
        return match_requests.data();
    }

    /** [UPDATE]
     * @brief Exchange Protocol with peer process.
     * @details This protocol exchange can be considered the first step in the matching
     * process. Note that MPI operations in this method use the tag of the original
     * communication call. While this should work for collectives (tag of 0 is used),
     * support for collectives is still considered experimental.
     * The communicator does not have to be ::MPIS_COMM_WORLD, as long as both sides
     * calling this method are aware of this. While it may seem like a valid idea to use
     * the MPI_Comm provided in the MPI communication call, it is not valid to do so.
     * Using that communicator may cause collisions with the user's own MPI
     * communications, and thus cause odd errors or deadlock in the program.
     */
    void start_protocol_exchange()
    {
        if (Operation::RECV == operation)
        {
            out("(R) Protocol Peer, Tag:", cw_peer, tag);
            check_mpi(MPI_Irecv(&protocol, sizeof(Protocol), MPI_BYTE, cw_peer, tag,
                                MPIS_COMM_PROTOCOL, &protocol_request));
        }
        else
        {
            out("(S) Protocol Peer, Tag:", cw_peer, tag);
            check_mpi(MPI_Isend(&protocol, sizeof(Protocol), MPI_BYTE, cw_peer, tag,
                                MPIS_COMM_PROTOCOL, &protocol_request));
        }
    }

    /** @brief Wait on the MPI communications exchanging Protocol information.*/
    void wait_on_protocol()
    {
        check_mpi(MPI_Wait(&protocol_request, MPI_STATUS_IGNORE));
        out("Final Protocol:", protocol);
    }

    /** [UPDATE]
     * @brief Aggregate any MPI communications for the matching phase into a vector.
     * @details This function asks a Request object to insert any MPI_Request objects that
     * were created as a part of initiating the matching phase into a larger list of MPI
     * communications. The input parameters are collecting MPI communications from a
     * series of Request objects in order to perform a single MPI_Waitall call on all of
     * them at once. This is done to avoid the deadlock caused by different processes
     * desiring to (validly) match requests in different orders. Explicitly does NOT set
     * Request::matched to true, but Request::match_requests and Request::match_statuses
     * are cleared.
     *
     * @param [in, out] request_train The vector of MPI_Request objects to append any
     * local matching-phase MPI_Requests to.
     */
    void join_waitall_match(std::vector<MPI_Request>& request_train)
    {
        request_train.insert(request_train.end(), match_requests.begin(),
                             match_requests.end());
        match_requests.clear();
    }

    /** @brief Obtain the memory type held by the Request object.
     * @return The GPUMemoryType held by the Request object.
     */
    GPUMemoryType get_memory_type()
    {
        return memory_type;
    }

    /** @brief A method for checking if the communication will require a synchronization
     * of memory between the GPU and the CPU.
     * @return A boolean indicating if the CPU needs to synchronize memory with the GPU.
     */
    bool needs_gpu_flush()
    {
        return (GPUMemoryType::COARSE == memory_type) &&
               ((Operation::SEND == operation) || (Operation::RSEND == operation));
    }

    /**
     * @brief Custom wrapper of Print::out that adds a prefix to print Request::myID
     * first.
     * @tparam UseRanks A compile-time option to determine if the printout should be
     * prefaced by a printout of the rank doing the printing.
     * @tparam Args The parameter pack of the types of things to print
     * through
     * @param [in] args The set of items to print out.
     */
    template <bool UseRanks = true, typename... Args>
    void out(Args&&... args)
    {
        Print::out<UseRanks>("[Req:", myID, "]", std::forward<Args>(args)...);
    }

    /**
     * @brief Determine what the rank of process @p base_rank in @p base_comm is in @p
     * lookup_comm.
     * @param [in] base_rank The rank of a process to translate.
     * @param [in] base_comm The communicator of the base rank.
     * @param [in] lookup_comm The communicator to perform rank translation on.
     * @return The rank of process @p base_rank as it is in @p lookup_comm. If @p
     * base_rank is not in @p lookup_comm, MPI_UNDEFINED will be returned.
     */
    static inline int rankLookup(int base_rank, MPI_Comm base_comm, MPI_Comm lookup_comm)
    {
        MPI_Group base_group;
        force_mpi(MPI_Comm_group(base_comm, &base_group));
        MPI_Group lookup_group;
        force_mpi(MPI_Comm_group(lookup_comm, &lookup_group));
        int base_ranks[1]   = {base_rank};
        int lookup_ranks[1] = {-1};
        force_mpi(MPI_Group_translate_ranks(base_group, 1, base_ranks, lookup_group,
                                            lookup_ranks));
        force_mpi(MPI_Group_free(&base_group));
        force_mpi(MPI_Group_free(&lookup_group));
        Print::out("Started with rank:", base_rank, "ended up with", lookup_ranks[0]);
        return lookup_ranks[0];
    }

protected:
    /** @brief The unique ID of a Request object. */
    size_t myID;
    /** @brief What type of GPU buffer the Request has. */
    GPUMemoryType memory_type;
    /** @brief A specific MPI_Request used for exchanging protocol with matching peer. */
    MPI_Request protocol_request;
    /** @brief A vector that holds any MPI communications that happen during matching. */
    std::vector<MPI_Request> match_requests;

    /** @brief Static method for assigning IDs to Request objects.
     * @details IDs start at 1 and increment by 1 per call to this function.
     * @return A new ID value to assign to a Request.
     */
    static size_t assignID()
    {
        static size_t ID = 1;
        return ID++;
    }

private:
    /**
     * @brief Specialization of rankLookup that uses MPIS_COMM_WORLD as the lookup
     * communicator.
     * @param [in] lookup_peer The rank in Request::comm to lookup in MPIS_COMM_WORLD.
     * @return The rank of @p lookup_peer inside MPIS_COMM_WORLD (MPI_COMM_WORLD).
     */
    int resolve_comm_world(int lookup_peer)
    {
        return rankLookup(lookup_peer, comm, MPIS_COMM_WORLD);
    }

    /** @brief Debug method that prints out most of the variables in a Request object. */
    void print()
    {
        out("Operation:", operation, protocol, "GPU Buffer Type:", memory_type,
            " - attributes:\n\tBuffers:", send_buffer, recv_buffer, "\n\tCount", count,
            "\n\tType:", datatype, "(total bytes of buffer:", get_size_of_buffer(),
            ")\n\tPeer:", peer, "\n\tTag:", tag, "\n\tComm:", comm, "\n\tOp:", op);
    }
};

}  // namespace Communication
#endif
