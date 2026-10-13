#ifndef ST_ABSTRACT_MATCH
#define ST_ABSTRACT_MATCH

#include <array>
#include <cstdint>
#include <vector>

#ifdef USE_CXI
#include <rdma/fi_rma.h>  // For libfabric object types

#include "safety/gpu.hpp"  // For gpu object types
#endif

#include "abstract/request.hpp"
#include "misc/print.hpp"
#include "safety/mpi.hpp"

/**
 * @file match.hpp
 * @brief Header file containing various data exchanges used in the matching phase.
 */

/** @brief Namespace containing any code related to carrying out the matching phase.
 * @details Note that many matching algorithms will allocate space for MPI_Request objects
 * inside the Communication::Request being matched. Further, all matching algorithms must
 * be non-blocking in nature to help avoid deadlock when matching a set of valid, but
 * arbitrarily ordered communication operations, except for Match::Blank, which needs to
 * use an eager MPI_Send of zero bytes.
 * Matching will be completed when Communication::Queue::finalize_match() is called (or
 * more specifically, when Communication::Request::join_waitall_match() is called).
 * @sa Communication::Request::get_match_requests()
 */
namespace Match
{

/** @brief The algorithm for a simple "pseudo" match. No data is exchanged. */
class Blank
{
public:
    /**
     * @brief Match to another rank via communication synchronization.
     * @details Both sides of the communication call this matching algorithm function,
     * unlike everything else in Match::Protocol. The matching algorithm here is a simple
     * MPI_Irecv followed by an MPI_Send. nullptr addresses are used as the buffers, with
     * zero bytes as the size. This algorithm simply "matches" the processes via MPI
     * communication synchronization. Note that, technically, any ::Queue backend that
     * uses MPI to carry out communication will not be able to escape the fact that MPI
     * communication operations are required to match every time they are started.
     * @param [in, out] req The Communication::Request to match on.
     */
    static void match(Communication::Request& req)
    {
        MPI_Request* mpi_requests = req.get_match_requests(1);
        check_mpi(
            MPI_Irecv(nullptr, 0, MPI_BYTE, req.peer, req.tag, req.comm, mpi_requests));
        check_mpi(MPI_Send(nullptr, 0, MPI_BYTE, req.peer, req.tag, req.comm));
    }
};

/** @brief Matching algorithms that correspond to a specific protocol. */
namespace Protocol
{
#ifdef USE_CXI

/** @brief The number of MPI_Request objects Communication::Protocol::RNDV needs. */
static constexpr size_t RNDV_REQUESTS_TO_USE = 2;

/**
 * @brief The receiver's side of matching for Communication::Protocol::RNDV.
 * @details Currently allocates two requests (Match::Protocol::RNDV_REQUESTS_TO_USE) per
 * side. This side (for receiving operations) sends details about the receiver's buffer
 * and then awaits details of the sender's "clear-to-send" buffer. The objects being
 * exchanged are libfabric based objects. Each of the two actions is on separate
 * communicators so as to not collide with other ongoing matching operations.
 *
 * @param [in] user_buffer_details Pointer to details about the receiver's buffer, in the
 * form of a fi_rma_iov.
 * @param [out] cts_details Location to store information about the peer's "clear-to-send"
 * buffer, in the form of an fi_rma_ioc.
 * @param [in, out] req Communication::Request to be matched and to store matching
 * MPI_Requests into.
 * @param [in] phase_a Communication::MPIS_COMM_WORLD duplicate for Phase A
 * communication.
 * @param [in] phase_b Communication::MPIS_COMM_WORLD duplicate for Phase B
 * communication.
 * @sa Match::Protocol::sender_rndv
 */
static void receiver_rndv(struct fi_rma_iov* user_buffer_details,
                          struct fi_rma_ioc* cts_details, Communication::Request& req,
                          MPI_Comm phase_a, MPI_Comm phase_b)
{
    req.out("(Receiver RNDV) Matching with:", req.peer, "(", req.cw_peer, ") and tag",
            req.tag);
    MPI_Request* mpi_requests = req.get_match_requests(RNDV_REQUESTS_TO_USE);

    req.out("(Recv) Sending: ", user_buffer_details->addr, user_buffer_details->len,
            user_buffer_details->key);
    check_mpi(MPI_Isend(user_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, req.cw_peer,
                        req.tag, phase_a, &mpi_requests[0]));
    check_mpi(MPI_Irecv(cts_details, sizeof(fi_rma_ioc), MPI_BYTE, req.cw_peer, req.tag,
                        phase_b, &mpi_requests[1]));
}

/**
 * @brief The sender's side of matching for Communication::Protocol::RNDV.
 * @details Currently allocates two requests (Match::Protocol::RNDV_REQUESTS_TO_USE) per
 * side. This side (for sending operations) posts a receive for details about the
 * receiving side's buffer, then sends details about its "clear-to-send" buffer. The
 * objects being exchanged are libfabric based objects. Each of the two actions is on
 * separate communicators so as to not collide with other ongoing matching operations.
 *
 * @param [out] recv_buffer_details Location to store information about the peer's
 * receiver's buffer, in the form of a fi_rma_iov.
 * @param [in] cts_details Pointer to details about the sender's "clear-to-send" buffer,
 * in the form of a fi_rma_ioc.
 * @param [in, out] req Communication::Request to be matched and to store matching
 * MPI_Requests into.
 * @param [in] phase_a Communication::MPIS_COMM_WORLD duplicate for Phase A
 * communication.
 * @param [in] phase_b Communication::MPIS_COMM_WORLD duplicate for Phase B
 * communication.
 * @sa Match::Protocol::receiver_rndv
 */
static void sender_rndv(struct fi_rma_iov* recv_buffer_details,
                        struct fi_rma_ioc* cts_details, Communication::Request& req,
                        MPI_Comm phase_a, MPI_Comm phase_b)
{
    req.out("(Sender RNDV) Matching with:", req.peer, "(", req.cw_peer, ") and tag",
            req.tag);
    MPI_Request* mpi_requests = req.get_match_requests(RNDV_REQUESTS_TO_USE);

    check_mpi(MPI_Irecv(recv_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, req.cw_peer,
                        req.tag, phase_a, &mpi_requests[0]));

    req.out("(Send) Sending: ", cts_details->addr, cts_details->count, cts_details->key);
    check_mpi(MPI_Isend(cts_details, sizeof(fi_rma_ioc), MPI_BYTE, req.cw_peer, req.tag,
                        phase_b, &mpi_requests[1]));
}

/** @brief [TODO] */
static constexpr size_t EAGER_REQUESTS_TO_USE = 1;

/**
 * @brief [TODO]
 * @param [in] user_buffer_details
 * @param [in, out] req
 * @param [in] phase_a
 */
static void receiver_eager(struct fi_rma_iov*      user_buffer_details,
                           Communication::Request& req, MPI_Comm phase_a)
{
    req.out("(Receiver EAGER) Matching with:", req.peer, "(", req.cw_peer, ") and tag",
            req.tag);
    MPI_Request* mpi_requests = req.get_match_requests(EAGER_REQUESTS_TO_USE);

    req.out("(Recv) Sending: ", user_buffer_details->addr, user_buffer_details->len,
            user_buffer_details->key);
    check_mpi(MPI_Isend(user_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, req.cw_peer,
                        req.tag, phase_a, &mpi_requests[0]));
}

/**
 * @brief [TODO]
 * @param [out] recv_buffer_details
 * @param [in, out] req
 * @param [in] phase_a
 */
static void sender_eager(struct fi_rma_iov*      recv_buffer_details,
                         Communication::Request& req, MPI_Comm phase_a)
{
    req.out("(Sender EAGER) Matching with:", req.peer, "(", req.cw_peer, ") and tag",
            req.tag);
    MPI_Request* mpi_requests = req.get_match_requests(EAGER_REQUESTS_TO_USE);

    check_mpi(MPI_Irecv(recv_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, req.cw_peer,
                        req.tag, phase_a, &mpi_requests[0]));
}

/** @brief The number of MPI_Request objects Communication::Protocol::IPC needs.
 * @details This value is currently specific to the HIP-based IPC protocol that the
 * ::CXIQueue uses.
 */
static constexpr size_t HIP_IPC_REQUESTS_TO_USE = 1;

/** @brief Helper struct for sending multiple items in one MPI communication. */
struct IPCBundle
{
    /** @brief The HIP IPC handle to a memory allocation. */
    hipIpcMemHandle_t handle;
    /** @brief The offset into the memory allocation where the true address is. */
    uint64_t offset;
};

/**
 * @brief The receiver's side of matching for Communication::Protocol::IPC.
 * @details Sends two Match::Protocol::IPCBundle objects to the corresponding peer
 * process, in a single MPI call. Each of those objects describes a specific location.
 *
 * @param [in] ipc_data An array of two Match::Protocol::IPCBundle objects, where the
 * first represents details of the receive buffer and the second represents details about
 * the receiving side's completion buffer.
 * @param [in, out] req Communication::Request to be matched and to store matching
 * MPI_Requests into.
 * @param [in] comm Communication::MPIS_COMM_WORLD duplicate to use for matching.
 * @sa Match::Protocol::sender_hip_ipc
 */
static void receiver_hip_ipc(std::array<IPCBundle, 2>& ipc_data,
                             Communication::Request& req, MPI_Comm comm)
{
    // Currently EAGER only
    MPI_Request* mpi_requests = req.get_match_requests(HIP_IPC_REQUESTS_TO_USE);
    req.out("(Receiver HIP IPC) Matching with:", req.peer, "(", req.cw_peer, ") and tag",
            req.tag);
    check_mpi(MPI_Isend(ipc_data.data(), sizeof(IPCBundle) * 2, MPI_BYTE, req.cw_peer,
                        req.tag, comm, &mpi_requests[0]));
}

/**
 * @brief The sender's side of matching for Communication::Protocol::IPC.
 * @details Receives two Match::Protocol::IPCBundle objects describing the corresponding
 * peer process's receive and completion buffers, in a single MPI call. Each of those
 * objects describes a specific location.
 *
 * @param [out] ipc_data An array of two Match::Protocol::IPCBundle objects, where the
 * first represents details of the receive buffer and the second represents details about
 * the receiving side's completion buffer.
 * @param [in, out] req Communication::Request to be matched and to store matching
 * MPI_Requests into.
 * @param [in] comm Communication::MPIS_COMM_WORLD duplicate to use for matching.
 * @sa Match::Protocol::receiver_hip_ipc
 */
static void sender_hip_ipc(std::array<IPCBundle, 2>& ipc_data,
                           Communication::Request& req, MPI_Comm comm)
{
    // Currently EAGER only
    MPI_Request* mpi_requests = req.get_match_requests(HIP_IPC_REQUESTS_TO_USE);
    req.out("(Send HIP IPC) Matching with:", req.peer, "(", req.cw_peer, ") and tag",
            req.tag);
    check_mpi(MPI_Irecv(ipc_data.data(), sizeof(IPCBundle) * 2, MPI_BYTE, req.cw_peer,
                        req.tag, comm, &mpi_requests[0]));
}

/** @brief The number of MPI_Request objects Communication::Protocol::CREDIT needs. */
static constexpr size_t CREDIT_REQUESTS_TO_USE = 2;
/** @brief Details about a credit slot's "clear-to-send" buffer. */
using CreditBundle = fi_rma_ioc;
/** @brief Compound type describing the info for the set of "clear-to-send buffers".
 * @details Is an array of size Communication::MAX_CREDIT_SLACK.
 */
using CreditArray = std::array<CreditBundle, Communication::MAX_CREDIT_SLACK>;

/**
 * @brief The receiver's side of matching for Communication::Protocol::CREDIT.
 * @details Performs two communications: the first sends the details about the "recv
 * buffers" to the peer process; the second receives information about where to signal
 * readiness for each of the credit slots.
 *
 * Note that the credit buffers are a part of one big allocation while
 * the send side's "clear-to-send buffers" are each their own libfabric object. This is
 * because the clear-to-send buffers track remote writes to them, so each credit slot
 * needs to have a different libfabric counter (in current implementation).
 *
 * Exact details about how the Communication::Protocol::CREDIT works can be found in the
 * documentation for ::CXISendCredit and ::CXIRecvCredit.
 * @param [in] user_buffer_details Pointer to details about the credit buffers, in the
 * form of a fi_rma_iov.
 * @param [out] credit_details Location to store details about the credit protocol's
 * send-side "clear-to-send buffers".
 * @param [in, out] req Communication::Request to be matched and to store matching
 * MPI_Requests into.
 * @param [in] phase_a Communication::MPIS_COMM_WORLD duplicate for Phase A
 * communication.
 * @param [in] phase_b Communication::MPIS_COMM_WORLD duplicate for Phase B
 * communication.
 * @sa Match::Protocol::sender_credit
 */
static void receiver_credit(struct fi_rma_iov* user_buffer_details,
                            CreditArray& credit_details, Communication::Request& req,
                            MPI_Comm phase_a, MPI_Comm phase_b)
{
    MPI_Request* mpi_requests = req.get_match_requests(CREDIT_REQUESTS_TO_USE);
    req.out("(Receiver Credit) Matching with:", req.peer, "(", req.cw_peer, ") and tag",
            req.tag);

    req.out("Sending: ", user_buffer_details->addr, user_buffer_details->len,
            user_buffer_details->key);
    check_mpi(MPI_Isend(user_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, req.cw_peer,
                        req.tag, phase_a, &mpi_requests[0]));
    check_mpi(MPI_Irecv(credit_details.data(),
                        sizeof(CreditBundle) * Communication::MAX_CREDIT_SLACK, MPI_BYTE,
                        req.cw_peer, req.tag, phase_b, &mpi_requests[1]));
}

/**
 * @brief The sender's side of matching for Communication::Protocol::CREDIT.
 * @details Performs two communications: the first receives the details about the "recv
 * buffers" from the peer process; the second sends information about where to signal
 * readiness for each of the credit slots.
 *
 * Note that the receiving side's credit buffers are a part of one big allocation while
 * the send side's "clear-to-send buffers" are each their own libfabric object. This is
 * because the "clear-to-send buffers" track remote writes to them, so each credit slot
 * needs to have a different libfabric counter (in current implementation).
 *
 * Exact details about how the Communication::Protocol::CREDIT works can be found in the
 * documentation for ::CXISendCredit and ::CXIRecvCredit.
 *
 * @param [out] recv_buffer_details Location to store information about the peer's
 * receiver's credit buffers, in the form of a fi_rma_iov.
 * @param [in] credit_details Details about the credit protocol's "clear-to-send buffer"
 * locations.
 * @param [in, out] req Communication::Request to be matched and to store matching
 * MPI_Requests into.
 * @param [in] phase_a Communication::MPIS_COMM_WORLD duplicate for Phase A
 * communication.
 * @param [in] phase_b Communication::MPIS_COMM_WORLD duplicate for Phase B
 * communication.
 * @sa Match::Protocol::receiver_credit
 */
static void sender_credit(struct fi_rma_iov* recv_buffer_details,
                          CreditArray& credit_details, Communication::Request& req,
                          MPI_Comm phase_a, MPI_Comm phase_b)
{
    MPI_Request* mpi_requests = req.get_match_requests(CREDIT_REQUESTS_TO_USE);
    req.out("(Send Credit) Matching with:", req.peer, "(", req.cw_peer, ") and tag",
            req.tag);

    check_mpi(MPI_Irecv(recv_buffer_details, sizeof(fi_rma_iov), MPI_BYTE, req.cw_peer,
                        req.tag, phase_a, &mpi_requests[0]));

    check_mpi(MPI_Isend(credit_details.data(),
                        sizeof(CreditBundle) * Communication::MAX_CREDIT_SLACK, MPI_BYTE,
                        req.cw_peer, req.tag, phase_b, &mpi_requests[1]));
}

/** @brief The number of MPI_Request objects Communication::Protocol::SELF needs. */
static constexpr size_t SELF_REQUESTS_TO_USE = 2;

/** @brief Compound type describing the full amount of data exchanged during matching.
 * @details The first index is for the pointer to the receive buffer, the second index is
 * for the pointer to the completion buffer.
 */
using SelfBundle = std::array<void*, 2>;

/**
 * @brief The sender's side of matching for Communication::Protocol::SELF.
 * @details The "sending" side of the protocol for when a process communicates with
 * itself. While there are two MPI communications that occur in this protocol, only the
 * first sends data. The second is to make sure both sides are "ready" when the match is
 * done by having them synchronize. The data is being received on this side in an array of
 * two pointers. The first is the address of the buffer used for the receive operation,
 * and the second is the address of the completion buffer for the receive operation.
 * Raw pointers can be exchanged here since both sides are on the same process (and thus
 * have a shared address space).
 *
 * @param [out] dst_data Location to save the pointers to the receive and remote
 * completion buffers.
 * @param [in, out] req Communication::Request to be matched and to store matching
 * MPI_Requests into.
 * @param [in] phase_a Communication::MPIS_COMM_WORLD duplicate for Phase A
 * communication.
 * @param [in] phase_b Communication::MPIS_COMM_WORLD duplicate for Phase B
 * communication.
 * @sa Match::Protocol::receiver_self
 */
static void sender_self(SelfBundle& dst_data, Communication::Request& req,
                        MPI_Comm phase_a, MPI_Comm phase_b)
{
    MPI_Request* mpi_requests = req.get_match_requests(SELF_REQUESTS_TO_USE);
    req.out("(Send Self Exchange) Matching with:", req.peer, "(", req.cw_peer,
            ") and tag", req.tag);

    check_mpi(MPI_Irecv(dst_data.data(), 2, MPI_AINT, req.cw_peer, req.tag, phase_a,
                        &mpi_requests[0]));
    check_mpi(
        MPI_Isend(nullptr, 0, MPI_BYTE, req.cw_peer, req.tag, phase_b, &mpi_requests[1]));
}

/**
 * @brief The receiver's side of matching for Communication::Protocol::SELF.
 * @details The "receiving" side of the protocol for when a process communicates with
 * itself. While there are two MPI communications that occur in this protocol, only the
 * first sends data. The second is to make sure both sides are "ready" when the match is
 * done by having them synchronize. The data is being sent in an array of
 * two pointers. The first is the address of the user buffer used in the receive
 * operation, and the second is the address of the completion buffer for the receive
 * operation. Raw pointers can be exchanged here since both sides are on the same process
 * (and thus have a shared address space).
 *
 * @param [in] dst_data Pointers to the receive and completion buffers.
 * @param [in, out] req Communication::Request to be matched and to store matching
 * MPI_Requests into.
 * @param [in] phase_a Communication::MPIS_COMM_WORLD duplicate for Phase A
 * communication.
 * @param [in] phase_b Communication::MPIS_COMM_WORLD duplicate for Phase B
 * communication.
 * @sa Match::Protocol::sender_self
 */
static void receiver_self(SelfBundle& dst_data, Communication::Request& req,
                          MPI_Comm phase_a, MPI_Comm phase_b)
{
    MPI_Request* mpi_requests = req.get_match_requests(SELF_REQUESTS_TO_USE);
    req.out("(Recv Self Exchange) Matching with:", req.peer, "(", req.cw_peer,
            ") and tag", req.tag);

    check_mpi(MPI_Isend(dst_data.data(), 2, MPI_AINT, req.cw_peer, req.tag, phase_a,
                        &mpi_requests[0]));
    check_mpi(
        MPI_Irecv(nullptr, 0, MPI_BYTE, req.cw_peer, req.tag, phase_b, &mpi_requests[1]));
}

#endif
};  // namespace Protocol

}  // namespace Match

#endif
