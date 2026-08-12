#ifndef ST_ABSTRACT_MATCH
#define ST_ABSTRACT_MATCH

#include <vector>

#ifdef USE_CXI
#include <rdma/fi_rma.h>  // For libfabric object types

#include "safety/gpu.hpp"  // For gpu object types
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
        check_mpi(
            MPI_Irecv(nullptr, 0, MPI_BYTE, req.peer, req.tag, req.comm, mpi_requests));
        check_mpi(MPI_Send(nullptr, 0, MPI_BYTE, req.peer, req.tag, req.comm));
    }
};

namespace ProtocolMatch
{
#ifdef USE_CXI

static constexpr size_t RNDV_REQUESTS_TO_USE = 2;

static void receiver_rndv(struct fi_rma_iov* user_buffer_details,
                          struct fi_rma_ioc* cts_details, Request& req, MPI_Comm phase_a,
                          MPI_Comm phase_b)
{
    int real_rank = req.resolve_comm_world();
    Print::out("(Receiver)", req.getID(), "Matching with:", real_rank, "(", req.peer,
               ") and tag", req.tag);
    MPI_Request* mpi_requests = req.get_match_requests(RNDV_REQUESTS_TO_USE);

    Print::out("(Recv) Sending: ", user_buffer_details->addr, user_buffer_details->len,
               user_buffer_details->key);
    check_mpi(MPI_Isend(user_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                        req.tag, phase_a, &mpi_requests[0]));
    check_mpi(MPI_Irecv(cts_details, sizeof(fi_rma_ioc), MPI_BYTE, real_rank, req.tag,
                        phase_b, &mpi_requests[1]));
}

static void sender_rndv(struct fi_rma_iov* recv_buffer_details,
                        struct fi_rma_ioc* cts_details, Request& req, MPI_Comm phase_a,
                        MPI_Comm phase_b)
{
    int real_rank = req.resolve_comm_world();
    Print::out("(Send)", req.getID(), "Matching with:", real_rank, "(", req.peer,
               ") and tag", req.tag);
    MPI_Request* mpi_requests = req.get_match_requests(RNDV_REQUESTS_TO_USE);

    check_mpi(MPI_Irecv(recv_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                        req.tag, phase_a, &mpi_requests[0]));

    Print::out("(Send) Sending: ", cts_details->addr, cts_details->count,
               cts_details->key);
    check_mpi(MPI_Isend(cts_details, sizeof(fi_rma_ioc), MPI_BYTE, real_rank, req.tag,
                        phase_b, &mpi_requests[1]));
}

static constexpr size_t HIP_IPC_REQUESTS_TO_USE = 1;

struct IPCBundle
{
    hipIpcMemHandle_t handle;
    uint64_t          offset;
};

static void receiver_hip_ipc(std::array<IPCBundle, 2>& ipc_data, Request& req,
                             MPI_Comm comm)
{
    // Currently EAGER only
    int          real_rank    = req.resolve_comm_world();
    MPI_Request* mpi_requests = req.get_match_requests(HIP_IPC_REQUESTS_TO_USE);
    check_mpi(MPI_Isend(ipc_data.data(), sizeof(IPCBundle) * 2, MPI_BYTE, real_rank,
                        req.tag, comm, &mpi_requests[0]));
}

static void sender_hip_ipc(std::array<IPCBundle, 2>& ipc_data, Request& req,
                           MPI_Comm comm)
{
    // Currently EAGER only
    int          real_rank    = req.resolve_comm_world();
    MPI_Request* mpi_requests = req.get_match_requests(HIP_IPC_REQUESTS_TO_USE);
    check_mpi(MPI_Irecv(ipc_data.data(), sizeof(IPCBundle) * 2, MPI_BYTE, real_rank,
                        req.tag, comm, &mpi_requests[0]));
}

static constexpr size_t CREDIT_REQUESTS_TO_USE = 2;
using CreditBundle                             = fi_rma_ioc;

static void sender_credit(struct fi_rma_iov*                          recv_buffer_details,
                          std::array<CreditBundle, MAX_CREDIT_SLACK>& credit_details,
                          Request& req, MPI_Comm phase_a, MPI_Comm phase_b)
{
    int          real_rank    = req.resolve_comm_world();
    MPI_Request* mpi_requests = req.get_match_requests(HIP_IPC_REQUESTS_TO_USE);

    Print::out("(Send Credit)", req.getID(), "Matching with:", real_rank, "(", req.peer,
               ") and tag", req.tag);

    check_mpi(MPI_Irecv(recv_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                        req.tag, phase_a, &mpi_requests[0]));

    check_mpi(MPI_Isend(credit_details.data(), sizeof(CreditBundle) * MAX_CREDIT_SLACK,
                        MPI_BYTE, real_rank, req.tag, phase_b, &mpi_requests[1]));
}

static void receiver_credit(struct fi_rma_iov* user_buffer_details,
                            std::array<CreditBundle, MAX_CREDIT_SLACK>& credit_details,
                            Request& req, MPI_Comm phase_a, MPI_Comm phase_b)
{
    int          real_rank    = req.resolve_comm_world();
    MPI_Request* mpi_requests = req.get_match_requests(RNDV_REQUESTS_TO_USE);
    Print::out("(Receiver Credit)", req.getID(), "Matching with:", real_rank, "(",
               req.peer, ") and tag", req.tag);

    Print::out("Sending: ", user_buffer_details->addr, user_buffer_details->len,
               user_buffer_details->key);
    check_mpi(MPI_Isend(user_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, real_rank,
                        req.tag, phase_a, &mpi_requests[0]));
    check_mpi(MPI_Irecv(credit_details.data(), sizeof(CreditBundle) * MAX_CREDIT_SLACK,
                        MPI_BYTE, real_rank, req.tag, phase_b, &mpi_requests[1]));
}

static constexpr size_t SELF_REQUESTS_TO_USE = 2;

using SelfBundle = std::array<void*, 2>;

static void sender_self(SelfBundle& dst_data, Request& req, MPI_Comm phase_a, MPI_Comm phase_b)
{
    int          real_rank    = req.resolve_comm_world();
    MPI_Request* mpi_requests = req.get_match_requests(SELF_REQUESTS_TO_USE);
    Print::out("(Send Self Exchange)", req.getID(), "Matching with:", real_rank, "(",
               req.peer, ") and tag", req.tag);

    check_mpi(MPI_Irecv(dst_data.data(), 2, MPI_AINT, real_rank, req.tag, phase_a,
                        &mpi_requests[0]));
    check_mpi(
        MPI_Isend(nullptr, 0, MPI_BYTE, real_rank, req.tag, phase_b, &mpi_requests[1]));
}

static void receiver_self(SelfBundle& dst_data, Request& req, MPI_Comm phase_a, MPI_Comm phase_b)
{
    int          real_rank    = req.resolve_comm_world();
    MPI_Request* mpi_requests = req.get_match_requests(SELF_REQUESTS_TO_USE);
    Print::out("(Recv Self Exchange)", req.getID(), "Matching with:", real_rank, "(",
               req.peer, ") and tag", req.tag);

    check_mpi(MPI_Isend(dst_data.data(), 2, MPI_AINT, real_rank, req.tag, phase_a,
                        &mpi_requests[0]));
    check_mpi(
        MPI_Irecv(nullptr, 0, MPI_BYTE, real_rank, req.tag, phase_b, &mpi_requests[1]));
}

#endif
};  // namespace ProtocolMatch

}  // namespace Communication

#endif
