#include "queues/CXIQueue.hpp"

#include "safety/hip.hpp"
#include "safety/libfabric.hpp"

void CXIQueue::libfabric_setup()
{
    // Hints will be freed by helper call below
    struct fi_info* hints;
    hints = fi_allocinfo();

    // set our requirements for the providers
    hints->caps = FI_SHARED_AV | FI_RMA | FI_REMOTE_READ | FI_REMOTE_WRITE |
                  FI_RMA_EVENT | FI_MSG | FI_WRITE | FI_HMEM | FI_TRIGGER;
    hints->addr_format   = FI_ADDR_CXI;
    hints->ep_attr->type = FI_EP_RDM;  // Must use for connection-less
    hints->tx_attr->caps = FI_SHARED_AV | FI_RMA | FI_WRITE | FI_RMA_EVENT |
                           FI_MSG | FI_HMEM | FI_TRIGGER;
    hints->rx_attr->caps = FI_SHARED_AV | FI_RMA | FI_WRITE | FI_REMOTE_WRITE |
                           FI_RMA_EVENT | FI_MSG | FI_HMEM | FI_TRIGGER;
    hints->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED |
                                  FI_MR_RMA_EVENT | FI_MR_LOCAL | FI_MR_HMEM;
    hints->mode = FI_CONTEXT;

    // Get information on available providers given our requirements in
    // hints
    force_libfabric(fi_getinfo(FI_VERSION(1, 15), 0, 0, 0, hints, &fi));
    fi_freeinfo(hints);  // deallocate hints

    // Create the fabric from the provider information
    check_libfabric(fi_fabric(fi->fabric_attr, &fabric, 0));

    // Make Domain
    force_libfabric(fi_domain(fabric, fi, &domain, 0));

    // Make some completion queues!
    struct fi_cq_attr cq_attr = {};
    cq_attr.size              = fi->tx_attr->size;
    cq_attr.flags             = 0;
    cq_attr.format            = FI_CQ_FORMAT_DATA;
    cq_attr.wait_obj          = FI_WAIT_UNSPEC;
    cq_attr.signaling_vector  = 0;
    cq_attr.wait_cond         = FI_CQ_COND_NONE;
    cq_attr.wait_set          = 0;
    force_libfabric(fi_cq_open(domain, &cq_attr, &txcq, 0));

    cq_attr.size = fi->rx_attr->size;
    force_libfabric(fi_cq_open(domain, &cq_attr, &rxcq, 0));

    // Make Address Vector
    struct fi_av_attr av_attr = {};
    av_attr.type              = FI_AV_TABLE;
    av_attr.count             = 2;
    av_attr.flags             = FI_SYMMETRIC;
    force_libfabric(fi_av_open(domain, &av_attr, &av, 0));

    // Endpoint setup
    force_libfabric(fi_endpoint(domain, fi, &ep, 0));

    // Bind T/R Completion Queues to EP -- Note flags!
    check_libfabric(fi_ep_bind(ep, &(txcq)->fid, FI_TRANSMIT));
    check_libfabric(fi_ep_bind(ep, &(rxcq)->fid, FI_RECV));
    // Bind EP to AV before enabling
    check_libfabric(fi_ep_bind(ep, &(av)->fid, 0));
    check_libfabric(fi_enable(ep));

    // Get our "cxi address"
    size_t _array_size = name_array_len;
    check_libfabric(fi_getname(&(ep)->fid, name, &_array_size));

    // End out of band setup

    /* Make counters
    struct fi_cntr_attr cntr_attr = {
        .events   = FI_CNTR_EVENTS_COMP,
        .wait_obj = FI_WAIT_UNSPEC,  // for using libfabric waits (can also
                                     // use FI_WAIT_NONE for no waits)
    };
    check_libfabric(fi_cntr_open(domain, &cntr_attr, &send_ctr, NULL));
    check_libfabric(fi_cntr_open(domain, &cntr_attr, &recv_ctr, NULL));
    check_libfabric(
        fi_cntr_open(domain, &cntr_attr, &completion_counter, NULL));
    check_libfabric(
        fi_cntr_open(domain, &cntr_attr, &completion_counter2, NULL));*/

    // Bind the counters
    // check_libfabric(fi_ep_bind(ep, &(send_ctr)->fid, FI_SEND));
    // check_libfabric(fi_ep_bind(ep, &(recv_ctr)->fid, FI_RECV));
}

void CXIQueue::peer_setup(int rank)
{
    // Do a setup
    char othername[name_array_len];
    memset(othername, 0, name_array_len * sizeof(char));
    size_t other_len = name_array_len;

    int my_rank;
    force_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    if (my_rank > rank)
    {
        force_mpi(MPI_Send(name, name_array_len, MPI_CHAR, rank, OOB_TAG,
                           MPI_COMM_WORLD));
        force_mpi(MPI_Recv(othername, name_array_len, MPI_CHAR, rank, OOB_TAG,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    else
    {
        force_mpi(MPI_Recv(othername, name_array_len, MPI_CHAR, rank, OOB_TAG,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        force_mpi(MPI_Send(name, name_array_len, MPI_CHAR, rank, OOB_TAG,
                           MPI_COMM_WORLD));
    }

    // Try to connect to peer?
    fi_addr_t* partner = &peers[rank];
    check_libfabric(fi_av_insert(av, othername, 1, partner, 0, 0));
}
