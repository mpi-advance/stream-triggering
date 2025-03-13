#ifndef ST_ABSTRACT_MATCH
#define ST_ABSTRACT_MATCH

#include <vector>

#include "safety/mpi.hpp"

namespace Communication
{

class BlankMatch
{
public:
    static constexpr int MATCH_TAG = 4200;

    static void match(int peer_rank, MPI_Comm comm = MPI_COMM_WORLD)
    {
        check_mpi(MPI_Sendrecv(nullptr, 0, MPI_BYTE, peer_rank, MATCH_TAG,
                               nullptr, 0, MPI_BYTE, peer_rank, MATCH_TAG, comm,
                               MPI_STATUS_IGNORE));
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
#ifdef USE_CUDA
        throw std::runtime_error("Type not supported for matching!");
#else
        static_assert(false, "Type not supported!");
#endif
    }
}

class ExchangeMatch
{
public:
    static constexpr int MATCH_TAG = 4211;

    template <typename T>
    static std::vector<T> match(std::vector<T>& data_to_exchange, int peer_rank,
                                MPI_Comm comm = MPI_COMM_WORLD)
    {
        std::vector<T> data_to_recv(data_to_exchange.size());

        MPI_Datatype my_type = type_to_use<T>();

        check_mpi(MPI_Sendrecv(
            data_to_exchange.data(), data_to_exchange.size(), my_type,
            peer_rank, MATCH_TAG, data_to_recv.data(), data_to_recv.size(),
            my_type, peer_rank, MATCH_TAG, comm, MPI_STATUS_IGNORE));

        return data_to_recv;
    }
};

class OneSideMatch
{
public:
    static constexpr int MATCH_TAG = 4222;

    template <typename T>
    static void give(std::vector<T>& data_to_exchange, int peer_rank,
              MPI_Comm comm = MPI_COMM_WORLD)
    {
        MPI_Datatype my_type = type_to_use<T>();

        check_mpi(MPI_Ssend(data_to_exchange.data(), data_to_exchange.size(),
                            my_type, peer_rank, MATCH_TAG, comm));
    }

    template <typename T>
    static std::vector<T> take(size_t num_elems, int peer_rank,
                        MPI_Comm comm = MPI_COMM_WORLD)
    {
        std::vector<T> recv_data(num_elems);
        MPI_Datatype   my_type = type_to_use<T>();

        check_mpi(MPI_Recv(recv_data.data(), recv_data.size(), my_type,
                           peer_rank, MATCH_TAG, comm, MPI_STATUS_IGNORE));
        return recv_data;
    }
};

}  // namespace Communication

#endif
