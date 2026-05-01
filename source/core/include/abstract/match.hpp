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
    static void match(Request& req)
    {
        MPI_Request* mpi_requests = req.get_match_requests(1);
        check_mpi(MPI_Irecv(nullptr, 0, MPI_BYTE, req.peer, req.tag, req.comm, mpi_requests));
        check_mpi(MPI_Send(nullptr, 0, MPI_BYTE, req.peer, req.tag, req.comm));
    }
};

#ifdef USE_CXI

class ProtocolMatch
{
public:
    static void receiver(struct fi_rma_iov* user_buffer_details, Operation* op_details,
                         struct fi_rma_ioc* cts_details, Request& req, MPI_Comm phase_a,
                         MPI_Comm phase_b)
    {
        int real_rank = req.resolve_comm_world();
        Print::out("(Receiver)", req.getID(), "Matching with:", real_rank, "(", req.peer,
                   ") and tag", req.tag);
        MPI_Request* mpi_requests = req.get_match_requests(REQUESTS_TO_USE);

        Print::out("(Recv) Sending: ", user_buffer_details->addr,
                   user_buffer_details->len, user_buffer_details->key);
        check_mpi(MPI_Isend(user_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                            req.tag, phase_a, &mpi_requests[0]));
        check_mpi(MPI_Irecv(op_details, sizeof(Operation), MPI_BYTE, real_rank, req.tag,
                            phase_b, &mpi_requests[1]));
        check_mpi(MPI_Irecv(cts_details, sizeof(fi_rma_ioc), MPI_BYTE, real_rank, req.tag,
                            phase_b, &mpi_requests[2]));
    }

    static void sender(struct fi_rma_iov* recv_buffer_details,
                       struct fi_rma_ioc* cts_details, Request& req, MPI_Comm phase_a,
                       MPI_Comm phase_b)
    {
        int real_rank = req.resolve_comm_world();
        Print::out("(Send)", req.getID(), "Matching with:", real_rank, "(", req.peer,
                   ") and tag", req.tag);
        MPI_Request* mpi_requests = req.get_match_requests(REQUESTS_TO_USE);

        check_mpi(MPI_Irecv(recv_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                            req.tag, phase_a, &mpi_requests[0]));

        Print::out("(Send) Sending: ", req.operation, cts_details->addr,
                   cts_details->count, cts_details->key);
        check_mpi(MPI_Isend(&req.operation, sizeof(Operation), MPI_BYTE, real_rank,
                            req.tag, phase_b, &mpi_requests[1]));
        check_mpi(MPI_Isend(cts_details, sizeof(fi_rma_ioc), MPI_BYTE, real_rank, req.tag,
                            phase_b, &mpi_requests[2]));
    }

    static constexpr size_t REQUESTS_TO_USE = 3;
};
#endif

}  // namespace Communication

#endif
