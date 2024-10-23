
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

class CXIObjects
{
public:
    using MMIOAddr = std::tuple<void*, size_t>;

    CXIObjects(struct fid_mr* mem_region, struct fid_cntr* trigger,
               struct fid_cntr* completion, struct fid_ep* main_ep)
        : my_mr(mem_region),
          trigger_cntr(trigger),
          completion_cntr(completion),
          trigger_internals(trigger),
          completion_internals(completion)
    {
        rma_work_entry      = {};
        message_description = {};
        msg_iov             = {};
        msg_rma_iov         = {};

        rma_work_entry.threshold       = 0;
        rma_work_entry.triggering_cntr = trigger;
        rma_work_entry.completion_cntr = completion;
        rma_work_entry.op_type         = FI_OP_WRITE;
        rma_work_entry.op.rma          = &message_description;

        message_description.ep                = main_ep;
        message_description.msg.msg_iov       = &msg_iov;
        message_description.msg.iov_count     = 1;
        message_description.msg.rma_iov       = &msg_rma_iov;
        message_description.msg.rma_iov_count = 1;
    }
    ~CXIObjects()
    {
        // Free MR if present
        if (my_mr)
        {
            force_libfabric(fi_close(&(my_mr)->fid));
        }
        // Free Counters
        force_libfabric(fi_close(&trigger_cntr->fid));
        force_libfabric(fi_close(&completion_cntr->fid));
    }

    void fill_message(fi_addr_t peer, void* buffer, size_t size)
    {
        msg_rma_iov                  = {0, size, get_mr_key()};
        msg_iov                      = {buffer, size};
        message_description.msg.addr = peer;
    }

    void bump_threshold()
    {
        rma_work_entry.threshold++;
    }

    size_t get_threshold()
    {
        return rma_work_entry.threshold;
    }

    void queue_work(struct fid_domain* domain)
    {
        force_libfabric(
            fi_control(&domain->fid, FI_QUEUE_WORK, &rma_work_entry));
    }

    size_t get_mr_key()
    {
        if (my_mr)
        {
            return fi_mr_key(my_mr);
        }
        else
        {
            return 0;
        }
    }

    MMIOAddr get_host_mmio_addr()
    {
        return std::make_pair(trigger_internals.mmio_addr,
                              trigger_internals.mmio_addr_len);
    }

    void* get_gpu_mmio_addr()
    {
        return trigger_internals.gpu_mmio_addr;
    }

    void* get_wb_buffer()
    {
        return completion_internals.wb_buffer;
    }

    void* get_gpu_wb_buffer()
    {
        return completion_internals.gpu_wb_buffer;
    }

private:
    enum CounterFlavor
    {
        Triggerable = 1,
        Complete    = 2,
        Both        = 3
    };
    template <CounterFlavor MODE>
    class CXICounter
    {
    public:
        CXICounter(struct fid_cntr* ctr) : counter(ctr)
        {
            // Open (create) CXI Extension object
            check_libfabric(fi_open_ops(&(ctr->fid), FI_CXI_COUNTER_OPS, 0,
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
                force_hip(
                    hipHostGetDevicePointer(&gpu_mmio_addr, mmio_addr, 0));
            }

            if constexpr (MODE & CounterFlavor::Complete)
            {
                // Allocate some memory with HIP for the counter to writeback
                // into
                force_hip(hipHostMalloc(&wb_buffer, sizeof(uint64_t),
                                        hipHostMallocDefault));
                // Set the writeback location to be this buffer
                check_libfabric(counter_ops->set_wb_buffer(
                    &counter->fid, wb_buffer, sizeof(uint64_t)));
                // Get GPU version of WB buffer
                force_hip(
                    hipHostGetDevicePointer(&gpu_wb_buffer, wb_buffer, 0));
            }
        }

        ~CXICounter()
        {
            if constexpr (MODE & CounterFlavor::Triggerable)
            {
                force_hip(hipHostUnregister(mmio_addr));
            }

            if constexpr (MODE & CounterFlavor::Complete)
            {
                force_hip(hipHostFree(wb_buffer));
            }
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

    // Allocated Libfabric Objects
    struct fid_mr*                         my_mr;
    struct fid_cntr*                       trigger_cntr;
    CXICounter<CounterFlavor::Triggerable> trigger_internals;
    struct fid_cntr*                       completion_cntr;
    CXICounter<CounterFlavor::Complete>    completion_internals;

    // Structs for the RMA DFWQ Entry:
    struct fi_deferred_work rma_work_entry;
    struct fi_op_rma        message_description;
    struct iovec            msg_iov;
    struct fi_rma_iov       msg_rma_iov;
};

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

    ~CXIQueue()
    {
        libfabric_teardown();
    }

    void enqueue_operation(std::shared_ptr<Request> qe) override
    {
        CXIObjects& cxi_stuff = request_map.at(qe->getID());
        cxi_stuff.bump_threshold();
        // CXI Start
        cxi_stuff.queue_work(domain);
        // GPU Start
        gpu_launch(cxi_stuff);
        // Keep track of active stuff
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

        uint64_t key = prepare_cxi_mr_key(*qe);
        qe->match(key);
    }

private:
    void             libfabric_setup();
    void             peer_setup(int rank);
    uint64_t         prepare_cxi_mr_key(Request&);
    void             libfabric_teardown();
    struct fid_cntr* alloc_counter();
    void             gpu_launch(CXIObjects&);

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

    // MR Management
    static size_t getMRID()
    {
        static size_t ID = 1;
        return ID++;
    }
    // Map of Request ID to CXIObject (counters, mr)
    std::map<size_t, CXIObjects> request_map;
    std::vector<size_t>          active_requests;

    // Hip Stream
    hipStream_t* the_stream;
};

#endif
