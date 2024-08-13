#ifndef ST_REQUEST_QUEUE
#define ST_REQUEST_QUEUE

#include "safety/mpi.hpp"

#include <tuple>

namespace Communication
{
enum Operation
{
	SEND,
	RECV
};

class MatchPair
{
public:
	MatchPair(bool &addr_to_mark, int peer_rank, Operation op)
	    : completion_addr(addr_to_mark), peer(peer_rank)
	{
		if(Operation::SEND == op)
		{
			check_mpi(MPI_Send_init(
			    &tmp_match_buffer, 1, MPI_INT, peer, MATCH_TAG, MPI_COMM_WORLD, &match_request));
		}
		else
		{
			check_mpi(MPI_Recv_init(
			    &tmp_match_buffer, 1, MPI_INT, peer, MATCH_TAG, MPI_COMM_WORLD, &match_request));
		}
	}

	void start()
	{
		check_mpi(MPI_Start(&match_request));
	}

	void wait()
	{
		check_mpi(MPI_Wait(&match_request, MPI_STATUS_IGNORE));
		completion_addr = true;
	}

protected:
	static const int MATCH_TAG = 12777;
	bool            &completion_addr;
	int              peer;
	int              tmp_match_buffer = 0;
	MPI_Request      match_request;
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
	      tag(_tag), comm(_comm), info(_info), match_pair(isMatched, _peer, _operation) {};

	bool is_matched()
	{
		return isMatched;
	}

	bool is_ready()
	{
		return isReady;
	}

	MatchPair& match()
	{
		match_pair.start();
		return match_pair;
	}

	void ready()
	{
		isReady = true;
	}

	void prepare()
	{
		// Do stuff
	}

protected:
	MatchPair match_pair;
	bool isReady   = false;
	bool isMatched = false;
};

} // namespace Communication

#endif