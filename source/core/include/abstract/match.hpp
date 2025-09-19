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
                         struct fi_rma_iov* cts_details, Request& req)
    {
        Print::out("(Receiver) Matching with:", req.peer, "and tag", req.tag);
        MPI_Request* mpi_requests = req.get_match_requests(REQUESTS_TO_USE);

        check_mpi(MPI_Isend(user_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, req.peer,
                            req.tag, req.comm, &mpi_requests[0]));
        check_mpi(MPI_Isend(completion_details, sizeof(fi_rma_iov), MPI_BYTE, req.peer,
                            req.tag, req.comm, &mpi_requests[1]));
        check_mpi(MPI_Irecv(op_details, sizeof(Operation), MPI_BYTE, req.peer, req.tag,
                            req.comm, &mpi_requests[2]));
        check_mpi(MPI_Irecv(cts_details, sizeof(fi_rma_iov), MPI_BYTE, req.peer, req.tag,
                            req.comm, &mpi_requests[3]));
    }

    static void sender(struct fi_rma_iov* recv_buffer_details,
                       struct fi_rma_iov* completion_details,
                       struct fi_rma_iov* cts_details, Request& req)
    {
        Print::out("(Send) Matching with:", req.peer, "and tag", req.tag);
        MPI_Request* mpi_requests = req.get_match_requests(REQUESTS_TO_USE);

        check_mpi(MPI_Irecv(recv_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, req.peer,
                            req.tag, req.comm, &mpi_requests[0]));
        check_mpi(MPI_Irecv(completion_details, sizeof(fi_rma_iov), MPI_BYTE, req.peer,
                            req.tag, req.comm, &mpi_requests[1]));
        check_mpi(MPI_Isend(&req.operation, sizeof(Operation), MPI_BYTE, req.peer,
                            req.tag, req.comm, &mpi_requests[2]));
        check_mpi(MPI_Isend(cts_details, sizeof(fi_rma_iov), MPI_BYTE, req.peer, req.tag,
                            req.comm, &mpi_requests[3]));
    }

    static constexpr size_t REQUESTS_TO_USE = 4;
};
#endif

}  // namespace Communication

#endif
