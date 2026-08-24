#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include <string.h>

#include <vector>

#include "misc/print.hpp"
#include "safety/mpi.hpp"

namespace Communication
{

enum Operation : int
{
    SEND,
    RSEND,
    RECV,
    BARRIER,
    ALLREDUCE
};

enum Protocol : int
{
    NONE = 0,
    IPC,
    EAGER,
    CREDIT,
    RNDV,
    SELF,
};

constexpr size_t MAX_CREDIT_SLACK   = 5;
constexpr size_t CREDIT_SIZE_CUTOFF = 32768;

extern bool IPC_PROTOCOL_ENABLED;
extern bool CREDIT_PROTOCOL_ENABLED;

extern MPI_Comm MPIS_COMM_WORLD;

enum GPUMemoryType
{
    COARSE = 1,
    FINE   = 2,
};

class Request
{
public:
    Operation    operation;
    Protocol     protocol;
    void*        send_buffer;
    void*        recv_buffer;
    MPI_Count    count;
    MPI_Datatype datatype;
    int          peer;
    int          tag;
    MPI_Comm     comm;
    MPI_Info     info;
    MPI_Op       op;
    int          cw_peer;  // Peer's rank in MPI Comm world

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
          matched(false)
    {
        constexpr int     string_size = 100;
        char              info_key[]  = "mpi_memory_alloc_kinds";
        std::vector<char> value(string_size, 0);
        int               flag = 0;
        // Pre MPI-4.0
        if (MPI_INFO_NULL != _info)
        {
            force_mpi(MPI_Info_get(_info, info_key, string_size, value.data(), &flag));
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
    };

    bool is_matched()
    {
        return matched;
    }

    void set_match()
    {
        matched = true;
    }

    size_t getID()
    {
        return myID;
    }

    inline size_t get_size_of_buffer()
    {
        int size = -1;
        check_mpi(MPI_Type_size(datatype, &size));
        return (size_t)(size * count);
    }

    MPI_Request* get_match_requests(size_t num)
    {
        match_requests = std::vector<MPI_Request>(num, MPI_REQUEST_NULL);
        match_statuses = std::vector<MPI_Status>(num);
        return match_requests.data();
    }

    void start_protocol_exchange(MPI_Comm channel)
    {
        if (Operation::RECV == operation)
        {
            out("(R) Protocol Peer, Tag:", cw_peer, tag);
            check_mpi(MPI_Irecv(&protocol, sizeof(Protocol), MPI_BYTE, cw_peer, tag, channel,
                                &protocol_request));
        }
        else
        {
            out("(S) Protocol Peer, Tag:", cw_peer, tag);
            check_mpi(MPI_Isend(&protocol, sizeof(Protocol), MPI_BYTE, cw_peer, tag, channel,
                                &protocol_request));
        }
    }

    void wait_on_protcol()
    {
        check_mpi(MPI_Wait(&protocol_request, MPI_STATUS_IGNORE));
        out("Final Protocol:", protocol);
    }

    void wait_on_match()
    {
        check_mpi(MPI_Waitall(match_requests.size(), match_requests.data(),
                              match_statuses.data()));
        matched = true;
    }

    /* Explicity does NOT set "matched = true" */
    void join_waitall_match(std::vector<MPI_Request>& request_train,
                            std::vector<MPI_Status>&  status_train)
    {
        request_train.insert(request_train.end(), match_requests.begin(),
                             match_requests.end());
        match_requests.clear();
        status_train.insert(status_train.end(), match_statuses.begin(),
                            match_statuses.end());
        match_statuses.clear();
    }

    GPUMemoryType get_memory_type()
    {
        return memory_type;
    }

    bool needs_gpu_flush()
    {
        return (GPUMemoryType::COARSE == memory_type) &&
               ((Operation::SEND == operation) || (Operation::RSEND == operation));
    }

    // Helper debug methods
    template <bool UseRanks = true, typename... Args>
    void out(Args&&... args)
    {
        Print::out<UseRanks>("[Req:", myID, "]", std::forward<Args>(args)...);
    }

    // Figure out "base_rank"'s rank in "lookup_comm"
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
    size_t                   myID;
    GPUMemoryType            memory_type;
    MPI_Request              protocol_request;
    std::vector<MPI_Request> match_requests;
    std::vector<MPI_Status>  match_statuses;
    bool                     matched = false;

    static size_t assignID()
    {
        static size_t ID = 1;
        return ID++;
    }

private:
    int resolve_comm_world(int lookup_peer)
    {
        return rankLookup(lookup_peer, comm, MPIS_COMM_WORLD);
    }

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
