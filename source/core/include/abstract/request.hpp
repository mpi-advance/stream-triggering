#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include <vector>
#include <string.h>

#include "misc/print.hpp"
#include "safety/mpi.hpp"

namespace Communication
{

enum Operation
{
    SEND,
    RECV,
    BARRIER
};

enum GPUMemoryType
{
    COARSE = 1,
    FINE   = 2,
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
          matched(false)
    {
        int size = -1;
        check_mpi(MPI_Type_size(_datatype, &size));
        Print::out("Request made with address, size, count, tag, and ID:", _buffer, size,
                   _count, tag, myID);

        constexpr int string_size = 100;
        char          info_key[]  = "mpi_memory_alloc_kinds";
        char          value[string_size];
        int           flag = 0;
        // Pre MPI-4.0
        if (MPI_INFO_NULL != _info)
        {
            force_mpi(MPI_Info_get(_info, info_key, string_size, value, &flag));
        }

        if (0 == strcmp(value, "rocm:device:fine"))
        {
            Print::out("Using fine-grained memory!");
            memory_type = GPUMemoryType::FINE;
        }
        else
        {
            Print::out("Using coarse-grained memory!");
            memory_type = GPUMemoryType::COARSE;
        }
    };

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

    GPUMemoryType get_memory_type()
    {
        return memory_type;
    }

protected:
    size_t                   myID;
    GPUMemoryType            memory_type;
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