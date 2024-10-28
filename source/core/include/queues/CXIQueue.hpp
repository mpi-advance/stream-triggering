
#ifndef ST_THREAD_QUEUE
#define ST_THREAD_QUEUE

#include <hip/hip_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_trigger.h>
// clang-format off
#include <rdma/fi_cxi_ext.h>
// clang-format on

#include <map>
#include <vector>

#include "abstract/queue.hpp"
#include "safety/hip.hpp"
#include "safety/libfabric.hpp"
#include "safety/mpi.hpp"

enum CounterFlavor
{
    Triggerable = 1,
    Completable = 2,
    Both        = 3
};

template <CounterFlavor MODE>
class CXICounter
{
public:
    static struct fid_cntr* alloc_counter(struct fid_domain* domain)
    {
        struct fid_cntr* new_ctr;
        // Make counters
        struct fi_cntr_attr cntr_attr = {
            .events   = FI_CNTR_EVENTS_COMP,
            .wait_obj = FI_WAIT_UNSPEC,  // for using libfabric waits (can also
                                         // use FI_WAIT_NONE for no waits)
        };
        force_libfabric(fi_cntr_open(domain, &cntr_attr, &new_ctr, NULL));
        return new_ctr;
    }

    CXICounter(struct fid_domain* domain) : counter(alloc_counter(domain))
    {
        // Open (create) CXI Extension object
        check_libfabric(fi_open_ops(&(counter->fid), FI_CXI_COUNTER_OPS, 0,
                                    (void**)&counter_ops, NULL));
        if constexpr (MODE & CounterFlavor::Triggerable)
        {
            // Get the MMIO Address of the counter
            check_libfabric(counter_ops->get_mmio_addr(
                &counter->fid, &mmio_addr, &mmio_addr_len));
            // Register MMIO Address w/ HIP
            force_hip(hipHostRegister(mmio_addr, mmio_addr_len,
                                      hipHostRegisterDefault));
            // Get GPU version of MMIO address
            force_hip(hipHostGetDevicePointer(&gpu_mmio_addr, mmio_addr, 0));
        }

        if constexpr (MODE & CounterFlavor::Completable)
        {
            // Allocate some memory with HIP for the counter to writeback
            // into
            force_hip(hipHostMalloc(&wb_buffer, sizeof(uint64_t),
                                    hipHostMallocDefault));
            // Set the writeback location to be this buffer
            check_libfabric(counter_ops->set_wb_buffer(&counter->fid, wb_buffer,
                                                       sizeof(uint64_t)));
            // Get GPU version of WB buffer
            force_hip(hipHostGetDevicePointer(&gpu_wb_buffer, wb_buffer, 0));
        }
    }

    ~CXICounter()
    {
        // Free counter
        force_libfabric(fi_close(&counter->fid));

        if constexpr (MODE & CounterFlavor::Triggerable)
        {
            force_hip(hipHostUnregister(mmio_addr));
        }

        if constexpr (MODE & CounterFlavor::Completable)
        {
            force_hip(hipHostFree(wb_buffer));
        }
    }

    void print()
    {
        size_t value = fi_cntr_read(counter);
        std::cout << "Value: " << value << std::endl;
    }

    // Libfabric Structs
    struct fid_cntr*        counter;
    struct fi_cxi_cntr_ops* counter_ops;
    // MMIO Pointers
    void*  mmio_addr;
    size_t mmio_addr_len;
    void*  gpu_mmio_addr;
    // WB Pointers
    void* wb_buffer;
    void* gpu_wb_buffer;
};

class CXIRequest
{
public:
    virtual ~CXIRequest() = default;
    virtual void start(hipStream_t* the_stream)
    {
        threshold++;
        start_host();
        start_gpu(the_stream);
    }
    virtual void wait_gpu(hipStream_t* the_stream) = 0;
    virtual void wait_host() = 0;

protected:
    virtual void start_host()                       = 0;
    virtual void start_gpu(hipStream_t* the_stream) = 0;

    size_t threshold = 0;
};

class CXIWait : virtual public CXIRequest
{
public:
    CXIWait(struct fid_domain* domain) : completion_counter(domain) {}

    void wait_gpu(hipStream_t* the_stream) override;
    void wait_host() override;

    struct fid_cntr* get_libfabric_counter()
    {
        return completion_counter.counter;
    }

    void print_counter()
    {
        completion_counter.print();
    }

private:
    CXICounter<CounterFlavor::Completable> completion_counter;
};

enum GPUMemoryType
{
    COARSE = 1,
    FINE = 2,
};
template<GPUMemoryType G>
class CXITrigger : virtual public CXIRequest
{
public:
    CXITrigger(struct fid_domain* domain) : trigger_counter(domain) {}

    void start_gpu(hipStream_t* the_stream) override;

    struct fid_cntr* get_libfabric_counter()
    {
        return trigger_counter.counter;
    }

    void print_counter()
    {
        trigger_counter.print();
    }

private:
    CXICounter<CounterFlavor::Triggerable> trigger_counter;
};

enum CommunicationType
{
    ONE_SIDED = 1,
    TWO_SIDED = 2,
};

template <CommunicationType MODE, GPUMemoryType G>
class CXISend : public CXITrigger<G>, public CXIWait
{
    using FI_DFWQ_TYPE =
        std::conditional_t<MODE == CommunicationType::ONE_SIDED,
                           struct fi_op_rma, struct fi_op_msg>;

public:
    CXISend(struct fid_domain* domain, struct fid_ep* main_ep)
        : CXITrigger<G>(domain), CXIWait(domain), domain_ptr(domain)
    {
        work_entry          = {};
        message_description = {};
        msg_iov             = {};
        msg_rma_iov         = {};

        work_entry.threshold       = 0;
        work_entry.triggering_cntr = CXITrigger<G>::get_libfabric_counter();
        work_entry.completion_cntr = CXIWait::get_libfabric_counter();
        if constexpr (CommunicationType::ONE_SIDED == MODE)
        {
            work_entry.op_type = FI_OP_WRITE;
        }
        else
        {
            work_entry.op_type = FI_OP_SEND;
            // TODO
            static_assert(false, "TWO SIDED not implemented yet");
        }
        work_entry.op.rma = &message_description;

        message_description.ep            = main_ep;
        message_description.msg.msg_iov   = &msg_iov;
        message_description.msg.iov_count = 1;
        // No harm in doing this if mode is not one_sided
        message_description.msg.rma_iov       = &msg_rma_iov;
        message_description.msg.rma_iov_count = 1;
    }

    void start_host() override
    {
        work_entry.threshold = threshold;
        force_libfabric(
            fi_control(&domain_ptr->fid, FI_QUEUE_WORK, &work_entry));
    }

    void fill_message(fi_addr_t peer, void* buffer, size_t size,
                      uint64_t remote_key)
    {
        msg_rma_iov                  = {0, size, remote_key};
        msg_iov                      = {buffer, size};
        message_description.msg.addr = peer;
    }

private:
    // Structs for the DFWQ Entry:
    struct fi_deferred_work work_entry;
    FI_DFWQ_TYPE            message_description;
    struct iovec            msg_iov;
    // Always here, if if not always used.
    struct fi_rma_iov msg_rma_iov;
    // Keep the domain pointer for later use
    struct fid_domain* domain_ptr;
};

class CXIRecvOneSided : public CXIWait
{
public:
    CXIRecvOneSided(struct fid_domain* domain, struct fid_ep* main_ep,
                    void* buffer, size_t _total_size)
        : CXIWait(domain), total_size(_total_size)
    {
        size_t   mr_id              = getMRID();
        uint64_t recv_key_requested = mr_id;

        force_libfabric(fi_mr_reg(domain, buffer, total_size, FI_REMOTE_WRITE,
                                  0, recv_key_requested, FI_RMA_EVENT, &my_mr,
                                  NULL));
        force_libfabric(fi_mr_bind(my_mr, &(main_ep)->fid, 0));

        // Bind counter
        force_libfabric(fi_mr_bind(my_mr, &(get_libfabric_counter())->fid,
                                   FI_REMOTE_WRITE));
        // Enable MR
        force_libfabric(fi_mr_enable(my_mr));
    }

    ~CXIRecvOneSided()
    {
        // Free MR
        force_libfabric(fi_close(&(my_mr)->fid));
    }

    void start_host() override
    {
        // Do nothing
    }

    void start_gpu(hipStream_t* the_stream) override
    {
        // Do nothing
    }

    size_t get_mr_key()
    {
        return fi_mr_key(my_mr);
    }

private:
    // Allocated Libfabric Objects
    struct fid_mr* my_mr;
    size_t         total_size;

    // MR Management
    static size_t getMRID()
    {
        static size_t ID = 1;
        return ID++;
    }
};

class CXIQueue : public Queue
{
public:
    using CXIObjects = std::unique_ptr<CXIRequest>;

    CXIQueue(hipStream_t* stream_addr) : the_stream(stream_addr)
    {
        int size;
        force_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size));
        peers.resize(size, 0);
        libfabric_setup();
    }

    ~CXIQueue()
    {
        libfabric_teardown();
    }

    void enqueue_operation(std::shared_ptr<Request> qe) override
    {
        CXIObjects& cxi_stuff = request_map.at(qe->getID());
        cxi_stuff->start(the_stream);

        // Keep track of active requests
        active_requests.push_back(qe->getID());
    }
    void enqueue_prepare(std::shared_ptr<Request> qe) override {}

    void enqueue_waitall() override;

    void host_wait() override
    {
        force_hip(hipStreamSynchronize(*the_stream));
    }

    void match(std::shared_ptr<Request> qe) override
    {
        int peer = qe->peer;
        if (0 == peers.at(peer))
            peer_setup(peer);

        prepare_cxi_mr_key(*qe);
    }

private:
    void libfabric_setup();
    void peer_setup(int rank);
    void prepare_cxi_mr_key(Request&);
    void libfabric_teardown();

    // Persistent Libfabric objects
    struct fi_info*    fi;     /*!< Provider's data and features */
    struct fid_fabric* fabric; /*!< Represents the network */
    struct fid_domain* domain; /*!< A subsection of the network */
    struct fid_av*     av;     /*!< Address vector for connections */
    struct fid_ep*     ep;     /*!< An endpoint */
    struct fid_cq*     txcq;   /*!< The transmit completion queue */
    struct fid_cq*     rxcq;   /*!< The receive completion queue */

    // Peer information
    static constexpr int    OOB_TAG        = 1244;
    static constexpr size_t name_array_len = 100;
    char                    name[name_array_len];
    std::vector<fi_addr_t>  peers;

    // Map of Request ID to CXIObject (counters, mr)
    std::map<size_t, CXIObjects> request_map;
    std::vector<size_t>          active_requests;

    // Hip Stream
    hipStream_t* the_stream;
};

#endif
