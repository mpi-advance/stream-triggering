#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include "mpi.h"

#include <tuple>

namespace Communication
{
enum Operation
{
	SEND,
	RECV
};

class Request
{
public:
	Operation    operation;
	void        *buffer;
	MPI_Count    count;
	MPI_Datatype datatype;
	int          peer;
	int          tag;
	MPI_Comm     comm;
	MPI_Info     info;

	Request(Operation    _operation,
	        void        *_buffer,
	        MPI_Count    _count,
	        MPI_Datatype _datatype,
	        int          _peer,
	        int          _tag,
	        MPI_Comm     _comm,
	        MPI_Info     _info)
	    : operation(_operation), buffer(_buffer), count(_count), datatype(_datatype), peer(_peer),
	      tag(_tag), comm(_comm), info(_info){};

	bool is_matched()
	{
		return isMatched;
	}

    void match()
    {
       isMatched = true; 
    }

protected:
	bool isMatched = false;
};

} // namespace Communication

#endif