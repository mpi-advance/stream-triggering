  #ifndef ST_ABSTRACT_MATCH
#define ST_ABSTRACT_MATCH

#include <vector>

#include "abstract/request.hpp"
#include "misc/print.hpp"
#include "safety/mpi.hpp"

namespace Communication
{

/** @brief contains functions to match two requests between host and gpu
 * @details]
 *  Synchronizes requests by using non-blocking send and a blocking send on each side.
 *  The message size is 0, thus preventing the need to allocate buffers for data. 
 * 
*/
class BlankMatch
{
public:

	/** @brief contains functions to match two requests between host and gpu
	 * @details
	 *  Synchronizes requests by using non-blocking send and a blocking send on each side.
	 *  The message size is 0, thus preventing the need to allocate buffers for data. 
	 *  
	 * @param peer_rank rank of the other process to synchronize. 
	 * @param tag integer tag to match against. 
	 * @param request local request to match 
	 * @param comm MPI Communicator to use for context, default is MPI_COMM_WORLD. 
	 */
    static void match(int peer_rank, int tag, MPI_Request* request,
                      MPI_Comm comm = MPI_COMM_WORLD)
    {
        check_mpi(MPI_Irecv(nullptr, 0, MPI_BYTE, peer_rank, tag, comm, request));
        check_mpi(MPI_Send(nullptr, 0, MPI_BYTE, peer_rank, tag, comm));
    }
};

/** @brief constant expression to confirm the MPI_TYPE based on size of the type.  
 * @details \todo Is used by the one-sided functions below, why is this check necessary?
 * 
 */
template <typename T>
constexpr MPI_Datatype type_to_use()
{
    if constexpr (4 == sizeof(T))
    {
        return MPI_INT;
    }
    else if constexpr (8 == sizeof(T))
    {
        return MPI_LONG;
    }
    else
    {
#ifdef ADVANCED_CPP23
        static_assert(false, "Type not supported!");
#else
        throw std::runtime_error("Type not supported for matching!");
#endif
    }
}

/** @brief Contains static functions for synchronizing requests using one-way communication. 
 * @details 
 * Sending side calls give and uses Issend to confirm receipt before the request completes.
 * Receiving side calls take, posts receives and returns.
 * 
 */
class OneSideMatch
{
public:
	
	/** @brief Contains static functions for sending messages to synchronize requests. 
	 * @details 
	 * Uses 2 non-blocking sends to transfer data_to_excahnge to its peer. 
	 * The third send only completes when the recieving peer posts that it has recieved the buffer. 
	 * Since MPI messages are not overtaking this should ensure that the previous requests
	 * have been received as well. 
	 * @param [in] data_to_exchange tuple: key to write location, key of completion buffer, offset into completion buffer array 
	 * @param [in, out] req Request object to be synchronized. 
	 */
    template <typename T>
    static void give(std::vector<T*>& data_to_exchange, Request& req)
    {
        constexpr MPI_Datatype my_type = type_to_use<T>();

        MPI_Request* mpi_requests = req.get_match_requests(3);

        Print::out("(Recv) Matching with:", req.peer, "and tag", req.tag);
        check_mpi(MPI_Isend(data_to_exchange[0], 1, my_type, req.peer, req.tag, req.comm,
                            &mpi_requests[0]));
        check_mpi(MPI_Isend(data_to_exchange[1], 1, my_type, req.peer, req.tag, req.comm,
                            &mpi_requests[1]));
        check_mpi(MPI_Issend(data_to_exchange[2], 1, my_type, req.peer, req.tag, req.comm,
                             &mpi_requests[2]));
    }

	/** @brief Contains static functions for sending messages to synchronize requests. 
	 * @details 
	 * Uses 3 non-blocking recvs to catch the three messages from the give function. 
	 *
	 * @param [out] data_to_exchange tuple: key to write location, key of completion buffer, offset into completion buffer array 
	 * @param [in, out] req Request object to be synchronized. 
	 */
    template <typename T>
    static void take(std::vector<T*>& data_to_exchange, Request& req)
    {
        constexpr MPI_Datatype my_type = type_to_use<T>();

        MPI_Request* mpi_requests = req.get_match_requests(3);

        Print::out("(Send) Matching with:", req.peer, "and tag", req.tag);
        check_mpi(MPI_Irecv(data_to_exchange[0], 1, my_type, req.peer, req.tag, req.comm,
                            &mpi_requests[0]));
        check_mpi(MPI_Irecv(data_to_exchange[1], 1, my_type, req.peer, req.tag, req.comm,
                            &mpi_requests[1]));
        check_mpi(MPI_Irecv(data_to_exchange[2], 1, my_type, req.peer, req.tag, req.comm,
                            &mpi_requests[2]));
    }
};

}  // namespace Communication

#endif
