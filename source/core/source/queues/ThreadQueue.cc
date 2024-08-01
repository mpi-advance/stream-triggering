#include "queues/ThreadQueue.hpp"

#include "safety/mpi.hpp"

ThreadRequest::ThreadRequest(Communication::Request *req)
{
	switch(req->operation)
	{
		case Communication::Operation::SEND:
			check_mpi(MPI_Send_init(req->buffer,
			                        req->count,
			                        req->datatype,
			                        req->peer,
			                        req->tag,
			                        req->comm,
			                        &my_request));
			break;
		case Communication::Operation::RECV:
			check_mpi(MPI_Recv_init(req->buffer,
			                        req->count,
			                        req->datatype,
			                        req->peer,
			                        req->tag,
			                        req->comm,
			                        &my_request));
			break;
		default:
			throw std::runtime_error("Invalid Request");
			break;
	}
}

void ThreadRequest::start()
{
	check_mpi(MPI_Start(&my_request));
}

bool ThreadRequest::done()
{
	int value = 0;
	check_mpi(MPI_Test(&my_request, &value, MPI_STATUS_IGNORE));
	return value;
}

void ThreadRequest::progress()
{
	done();
}