#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include <unistd.h>

#include <optional>

#include "safety/mpi.hpp"

template <class>
inline constexpr bool always_false_v = false;

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

class MatchData
{
public:
    template <typename T>
    MatchData(T base_value) : data_len(sizeof(T))
    {
        my_match_data   = malloc(sizeof(T));
        peer_match_data = calloc(1, sizeof(T));
        T* temp_origin  = (T*)my_match_data;
        temp_origin[0]  = base_value;

        if constexpr (4 == sizeof(T))
        {
            my_type = MPI_INT;
        }
        else if constexpr (8 == sizeof(T))
        {
            my_type = MPI_LONG;
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

    ~MatchData()
    {
        free(my_match_data);
        free(peer_match_data);
    }

    MatchData(const MatchData&)            = delete;
    MatchData& operator=(const MatchData&) = delete;

    void* get_original_match_data()
    {
        return my_match_data;
    }

    void* get_peer_match_data() const
    {
        return peer_match_data;
    }

    size_t get_match_size()
    {
        return data_len;
    }

    bool is_matched()
    {
        return matched;
    }

    MPI_Datatype get_match_type()
    {
        return my_type;
    }

    void set_matched()
    {
        matched = true;
    }

private:
    void*  my_match_data;
    void*  peer_match_data;
    size_t data_len;
    bool   matched = false;

    MPI_Datatype my_type;
};

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
          myMatchData(std::nullopt),
          myID(assignID()) {};

    bool is_matched()
    {
        if (myMatchData)
        {
            return myMatchData->is_matched();
        }
        else
        {
            return false;
        }
    }

    bool is_ready()
    {
        return myReadyCheck.canProceed();
    }

    void match()
    {
        match(myID);
    }

    template <class T>
    void match(T thing)
    {
        matchMPI(peer, thing);
    }

    const std::optional<MatchData>& getMatch()
    {
        return myMatchData;
    }

    void ready()
    {
        myReadyCheck.readyUp();
    }

    size_t getID()
    {
        return myID;
    }

protected:
    std::optional<MatchData> myMatchData;
    ReadyCheck               myReadyCheck;
    size_t                   myID;

    static size_t assignID()
    {
        static size_t ID = 1;
        return ID++;
    }

    template <class T>
    void matchMPI(int peer_rank, T value)
    {
        myMatchData.emplace(value);

        MPI_Datatype type_to_use = myMatchData->get_match_type();
        void*        send_buf    = myMatchData->get_original_match_data();
        void*        recv_buf    = myMatchData->get_peer_match_data();

        int MATCH_TAG = 0;
        // TODO: Adjust MPI_COMM_WORLD
        MPI_Sendrecv(send_buf, 1, type_to_use, peer_rank, MATCH_TAG, recv_buf,
                     1, type_to_use, peer_rank, MATCH_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        myMatchData->set_matched();
    }
};

}  // namespace Communication
#endif