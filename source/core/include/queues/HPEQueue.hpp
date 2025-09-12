#ifndef ST_HPE_QUEUE
#define ST_HPE_QUEUE

#include "abstract/queue.hpp"
#include "mpi.h"
#include "safety/mpi.hpp"

#include <hip/hip_runtime.h>

/** @brief Derived class from QueueEntry to work with HPE systems
 * @details
 *	 includes overrides for the virtual functions in QueueEntry
 *   as well as flags and pointers for signaling between the cpu and gpu.
 * \todo REMOVE HPEQUEUE
 */
class HPEQueueEntry : public QueueEntry
{
public:
	HPEQueueEntry(MPI_Request req);
	~HPEQueueEntry();

	virtual void prepare() override;

	virtual void start() override;

	virtual bool done() override;

	virtual void progress() override;
};

/** @brief Version of HPEQueueEntry to use with point to point communication
 *  @details
 *     Direction is true if sending operation.
 *      
 */
template<bool Direction>
class HPEQueueEntryP2P : public HPEQueueEntry
{
public:
	HPEQueueEntryP2P(void        *_buffer,
	                 int          _count,
	                 MPI_Datatype _datatype,
	                 int          _peer,
	                 int          _tag,
	                 void        *_hpe_stream)
	    : HPEQueueEntry(MPI_REQUEST_NULL), buffer(_buffer), count(_count), datatype(_datatype),
	      peer(_peer), tag(_tag), hpe_stream((MPIS_Queue *) _hpe_stream)
	{
		prepare(); // Do we like this?
	};
	~HPEQueueEntryP2P();

    /** @brief
	 *  @details
	 * 
	 */
	void prepare() override
	{
		if constexpr(Direction)
		{
			force_mpi(
			    MPIS_Enqueue_send(buffer, count, datatype, peer, tag, *hpe_stream, &my_request));
		}
		else
		{
			force_mpi(
			    MPIS_Enqueue_recv(buffer, count, datatype, peer, tag, *hpe_stream, &my_request));
		}
	}

    /** @copydoc HPEQueueEntry
	 *  @details
	 * 		starts starts processing queue on stream. 
	 */
	void start() override
	{
		force_mpi(MPIS_Enqueue_start(*hpe_stream))
	}

	bool done() override;

	void progress() override;

private:
	void        *buffer;
	int          count;
	MPI_Datatype datatype;
	int          peer;
	int          tag;
	MPIS_Queue  *hpe_stream;
};

class HPEQueue : public Queue
{
public:
	HPEQueue(hipStream_t *stream_addr);
	~HPEQueue();

	QueueEntry *create_entry(MPI_Request) override;
	QueueEntry *create_send(const void *, int, MPI_Datatype, int, int) override;
	QueueEntry *create_recv(void *, int, MPI_Datatype, int, int) override;

	void enqueue_operation(QueueEntry *) override;
	void enqueue_waitall() override;
	void host_wait() override;

protected:
	hipStream_t *my_stream;

	MPI_Comm   dup_comm;
	MPIS_Queue my_queue;
};

#endif