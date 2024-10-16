
#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <hip/hip_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#include <vector>

#include "abstract/queue.hpp"
#include "safety/mpi.hpp"

class CXIQueue : public Queue
{
public:
    CXIQueue(hipStream_t* stream_addr) : the_stream(stream_addr)
    {
        int size;
        force_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size));
        peers.resize(size, 0);
        libfabric_setup();
    }
    void enqueue_operation(std::shared_ptr<Request> qe) override {}
    void enqueue_prepare(std::shared_ptr<Request> qe) override {}

    void enqueue_waitall() override {}

    void host_wait() override {}

    void match(std::shared_ptr<Request> qe) override {
        int peer = qe->peer;
        if(0 == peers.at(peer))
            peer_setup(peer);
    }

private:
    void libfabric_setup();
    void peer_setup(int rank);

    // Libfabric objects
    struct fi_info*    fi;     /*!< Provider's data and features */
    struct fid_fabric* fabric; /*!< Represents the network */
    struct fid_domain* domain; /*!< A subsection of the network */
    struct fid_av*     av;     /*!< Address vector for connections */
    struct fid_ep*     ep;     /*!< An endpoint */
    struct fid_cq*     txcq;   /*!< The transmit completion queue */
    struct fid_cq*     rxcq;   /*!< The receive completion queue */
    // struct fid_cntr*   send_ctr;
    // struct fid_cntr*   recv_ctr;
    // struct fid_cntr*   completion_counter;
    // struct fid_cntr*   completion_counter2;

    // Peer information
    static constexpr int    OOB_TAG        = 1244;
    static constexpr size_t name_array_len = 100;
    char                    name[name_array_len];
    std::vector<fi_addr_t>  peers;

    hipStream_t* the_stream;
};

#endif
