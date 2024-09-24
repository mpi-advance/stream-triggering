#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include <unistd.h>

#include <optional>

#include "safety/mpi.hpp"

namespace Communication
{

class ReadyCheck
{
public:
    ReadyCheck() : ready_counter(0) {}

    void readyUp()
    {
        ready_counter++;
    }

    void consume()
    {
        if (ready_counter > 0)
            ready_counter--;
    }

    bool canProceed()
    {
        return ready_counter > 0;
    }

protected:
    int ready_counter = 0;
};

enum Operation
{
    SEND,
    RECV
};

using MatchData = std::optional<int>;

class Request
{
public:
    Operation    operation;
    void*        buffer;
    MPI_Count    count;
    MPI_Datatype datatype;
    int          peer;
    int          tag;
    MPI_Comm     comm;
    MPI_Info     info;

    int ready_counter;
    int prep_counter;

    static const int NO_VALUE = -1;

    Request(Operation _operation, void* _buffer, MPI_Count _count,
            MPI_Datatype _datatype, int _peer, int _tag, MPI_Comm _comm,
            MPI_Info _info)
        : operation(_operation),
          buffer(_buffer),
          count(_count),
          datatype(_datatype),
          peer(_peer),
          tag(_tag),
          comm(_comm),
          info(_info),
          ready_counter(0),
          prep_counter(0),
          myID(assignID()) {};

    bool is_matched()
    {
        return isMatched;
    }

    bool is_ready()
    {
        return myReadyCheck.canProceed();
    }

    void match()
    {
        setMatch(matchMPI(peer, myID));
    }

    void setMatch(MatchData match_data)
    {
        myMatchData = match_data;
        isMatched   = true;
    }

    const MatchData& getMatch()
    {
        return myMatchData;
    }

    void ready()
    {
        myReadyCheck.readyUp();
    }

    void prepare()
    {
        static int BASE_PREPARE_TAG = 12888;
        int        exchange_value   = -1;
        int        recv_value       = -1;
        force_mpi(MPI_Sendrecv(&exchange_value, 1, MPI_INT, peer,
                               BASE_PREPARE_TAG, &recv_value, 1, MPI_INT, peer,
                               BASE_PREPARE_TAG, comm, MPI_STATUS_IGNORE));
        ready();
    }

    size_t getID()
    {
        return myID;
    }

protected:
    MatchData myMatchData;
    bool      isMatched = false;

    ReadyCheck myReadyCheck;
    size_t     myID;

    static size_t assignID()
    {
        static size_t ID = 0;
        return ID++;
    }

    MatchData matchMPI(int peer_rank, int value)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int MATCH_TAG = 0;
        if (value < -1)
            throw std::runtime_error("Exchange value cannot be negative!");

        int exchange_value = value;
        int recv_value     = -1;

        // TODO: Adjust MPI_COMM_WORLD
        MPI_Sendrecv(&exchange_value, 1, MPI_INT, peer_rank, MATCH_TAG,
                     &recv_value, 1, MPI_INT, peer_rank, MATCH_TAG,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (0 <= recv_value)
            return recv_value;
        else
            return std::nullopt;
    }
};

}  // namespace Communication
#endif