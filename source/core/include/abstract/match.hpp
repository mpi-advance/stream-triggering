#ifndef ST_ABSTRACT_MATCH
#define ST_ABSTRACT_MATCH

#include <vector>

#ifdef USE_CXI
#include <rdma/fi_rma.h>
#endif

#include "abstract/request.hpp"
#include "misc/print.hpp"
#include "safety/mpi.hpp"

namespace Communication
{

class BlankMatch
{
public:
    static void match(int peer_rank, int tag, MPI_Request* request,
                      MPI_Comm comm = MPI_COMM_WORLD)
    {
        check_mpi(MPI_Irecv(nullptr, 0, MPI_BYTE, peer_rank, tag, comm, request));
        check_mpi(MPI_Send(nullptr, 0, MPI_BYTE, peer_rank, tag, comm));
    }
};

#ifdef USE_CXI

class ProtocolMatch
{
public:
    static void receiver(struct fi_rma_iov* user_buffer_details,
                         struct fi_rma_iov* completion_details, Operation* op_details,
                         struct fi_rma_iov* cts_details, Request& req, MPI_Comm phase_a,
                         MPI_Comm phase_b)
    {
        int real_rank = rankLookup(req.peer, req.comm, phase_a);
        Print::out("(Receiver) Matching with:", real_rank, "and tag", req.tag);
        MPI_Request* mpi_requests = req.get_match_requests(REQUESTS_TO_USE);

        Print::out("(Recv) Sending: ", user_buffer_details->addr,
                   user_buffer_details->len, user_buffer_details->key,
                   completion_details->addr, completion_details->len,
                   completion_details->key);
        check_mpi(MPI_Isend(user_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                            req.tag, phase_a, &mpi_requests[0]));
        check_mpi(MPI_Isend(completion_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                            req.tag, phase_a, &mpi_requests[1]));
        check_mpi(MPI_Irecv(op_details, sizeof(Operation), MPI_BYTE, real_rank, req.tag,
                            phase_b, &mpi_requests[2]));
        check_mpi(MPI_Irecv(cts_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank, req.tag,
                            phase_b, &mpi_requests[3]));
    }

    static void sender(struct fi_rma_iov* recv_buffer_details,
                       struct fi_rma_iov* completion_details,
                       struct fi_rma_iov* cts_details, Request& req, MPI_Comm phase_a,
                       MPI_Comm phase_b)
    {
        int real_rank = rankLookup(req.peer, req.comm, phase_a);
        Print::out("(Send) Matching with:", real_rank, "and tag", req.tag);
        MPI_Request* mpi_requests = req.get_match_requests(REQUESTS_TO_USE);

        check_mpi(MPI_Irecv(recv_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                            req.tag, phase_a, &mpi_requests[0]));
        check_mpi(MPI_Irecv(completion_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                            req.tag, phase_a, &mpi_requests[1]));

        Print::out("(Send) Sending: ", req.operation, cts_details->addr, cts_details->len,
                   cts_details->key);
        check_mpi(MPI_Isend(&req.operation, sizeof(Operation), MPI_BYTE, real_rank,
                            req.tag, phase_b, &mpi_requests[2]));
        check_mpi(MPI_Isend(cts_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank, req.tag,
                            phase_b, &mpi_requests[3]));
    }

    // Figure out "base_rank"'s rank in "lookup_comm"
    static inline int rankLookup(int base_rank, MPI_Comm base_comm, MPI_Comm lookup_comm)
    {
        MPI_Group base_group;
        force_mpi(MPI_Comm_group(base_comm, &base_group));
        MPI_Group lookup_group;
        force_mpi(MPI_Comm_group(lookup_comm, &lookup_group));
        int base_ranks[1]  = {base_rank};
        int lookup_ranks[1] = {-1};
        force_mpi(MPI_Group_translate_ranks(base_group, 1, base_ranks, lookup_group,
                                            lookup_ranks));
        force_mpi(MPI_Group_free(&base_group));
        force_mpi(MPI_Group_free(&lookup_group));
        Print::out("Started with rank:", base_rank, " ended up with", lookup_ranks[0]);
        return lookup_ranks[0];
    }

    static constexpr size_t REQUESTS_TO_USE = 4;
};
#endif

}  // namespace Communication

#endif
