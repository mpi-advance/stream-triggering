#ifndef ST_ABSTRACT_MATCH
#define ST_ABSTRACT_MATCH

#include <vector>

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

class OneSideMatch
{
public:
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
