#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include "match.hpp"
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
          myID(assignID()) {};

    bool is_matched()
    {
        return matched;
    }

    bool is_ready()
    {
        return myReadyCheck.canProceed();
    }

    void toggle_match()
    {
        matched = true;
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
    ReadyCheck myReadyCheck;
    size_t     myID;
    bool       matched = false;

    static size_t assignID()
    {
        static size_t ID = 1;
        return ID++;
    }
};

}  // namespace Communication
#endif