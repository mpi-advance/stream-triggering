#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include <vector>

#include "safety/mpi.hpp"

namespace Communication
{

enum Operation
{
    SEND,
    RECV,
    BARRIER
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

    Request(Operation _operation, void* _buffer, MPI_Count _count, MPI_Datatype _datatype,
            int _peer, int _tag, MPI_Comm _comm, MPI_Info _info)
        : operation(_operation),
          buffer(_buffer),
          count(_count),
          datatype(_datatype),
          peer(_peer),
          tag(_tag),
          comm(_comm),
          info(_info),
          myID(assignID()),
          matched(false) {};

    bool is_matched()
    {
        return matched;
    }

    size_t getID()
    {
        return myID;
    }

    MPI_Request* get_match_requests(size_t num)
    {
        match_requests = std::vector<MPI_Request>(num, MPI_REQUEST_NULL);
        match_statuses = std::vector<MPI_Status>(num);
        return match_requests.data();
    }

    void wait_on_match()
    {
        check_mpi(MPI_Waitall(match_requests.size(), match_requests.data(),
                              match_statuses.data()));
        matched = true;
    }

protected:
    size_t                   myID;
    std::vector<MPI_Request> match_requests;
    std::vector<MPI_Status>  match_statuses;
    bool                     matched = false;

    static size_t assignID()
    {
        static size_t ID = 1;
        return ID++;
    }
};

}  // namespace Communication
#endif