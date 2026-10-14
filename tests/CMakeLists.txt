# Multi-backend tests, run through CTest. Each test is built once per backend
# variant the library was configured with, named <test>_<variant> and labeled
# <variant>:
#   ctest -L thread          # every test for one variant
#   ctest -R halo            # one test for every variant
# Tests run through launch.sh, which picks the MPI launcher at run time.

# The tests use GPU code (kernels, GPU buffers from MPIS_Alloc_mem) even for the
# thread backend, so they need a library built with a GPU backend. The GPU
# targets come from the top-level find_package calls.
if(USE_CUDA_BACKEND)
    set(ST_TEST_GPU CUDA)
    set(ST_TEST_GPU_LIBS CUDA::cudart CUDA::cuda_driver)
elseif(USE_HIP_BACKEND OR USE_CXI_BACKEND)
    set(ST_TEST_GPU HIP)
    set(ST_TEST_GPU_LIBS hip::host)
else()
    message(WARNING "The tests need a GPU backend (CUDA, HIP or CXI) enabled, skipping them.")
    return()
endif()
set(CMAKE_${ST_TEST_GPU}_STANDARD 20)

# Backend variants. Each defines <BACKEND>_BACKEND (as tests/compile.py does), and
# cxi-fine also defines FINE_GRAINED_TEST.
set(ST_TEST_VARIANTS "")
if(USE_CXI_BACKEND)
    list(APPEND ST_TEST_VARIANTS cxi-coarse cxi-fine)
endif()
foreach(backend hip cuda thread)
    string(TOUPPER ${backend} BACKEND)
    if(USE_${BACKEND}_BACKEND)
        list(APPEND ST_TEST_VARIANTS ${backend})
    endif()
endforeach()
message(STATUS "Test variants (${ST_TEST_GPU}): ${ST_TEST_VARIANTS}")

# st_add_test(<name> [RANKS <n>] [ARGS ...]) builds multi-backend/<name>.cpp.
# RANKS defaults to 2.
function(st_add_test name)
    cmake_parse_arguments(PARSE_ARGV 1 ST "" "RANKS" "ARGS")
    if(NOT ST_RANKS)
        set(ST_RANKS 2)
    endif()

    set(source multi-backend/${name}.cpp)
    set_source_files_properties(${source} PROPERTIES LANGUAGE ${ST_TEST_GPU})

    foreach(variant IN LISTS ST_TEST_VARIANTS)
        set(target ${name}_${variant})
        string(REGEX REPLACE "-.*" "" backend ${variant})
        string(TOUPPER ${backend}_BACKEND backend_define)

        add_executable(${target} ${source})
        target_compile_definitions(${target} PRIVATE NEED_${ST_TEST_GPU} ${backend_define}
            $<$<STREQUAL:${variant},cxi-fine>:FINE_GRAINED_TEST>)
        target_link_libraries(${target} PRIVATE stream-triggering MPI::MPI_CXX ${ST_TEST_GPU_LIBS})

        # The CXI variants take MPICH_GPU_SUPPORT_ENABLED from the job's environment
        set(env ST_DEFAULT_LAUNCHER=${MPIEXEC_EXECUTABLE} ST_DEFAULT_RANK_FLAG=${MPIEXEC_NUMPROC_FLAG})
        if(NOT backend STREQUAL cxi)
            list(APPEND env MPICH_GPU_SUPPORT_ENABLED=1)
        endif()

        add_test(NAME ${target}
                 COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/launch.sh ${ST_RANKS}
                         $<TARGET_FILE:${target}> ${ST_ARGS})
        set_tests_properties(${target} PROPERTIES LABELS ${variant} ENVIRONMENT "${env}")
    endforeach()
endfunction()

### Test catalog
st_add_test(allreduce)
st_add_test(as_nonblocking)
st_add_test(barrier)
st_add_test(duplicate_requests)
st_add_test(enqueue_wait_statuses)
st_add_test(enqueue_waitall_empty_statuses)
st_add_test(halo ARGS 10 64)
st_add_test(halo_and_allreduce ARGS 10 64)
st_add_test(hello_world)
st_add_test(invalid_request_state)
st_add_test(null_objects)
st_add_test(queue_test)
st_add_test(request_free_recreate)
st_add_test(ring)
st_add_test(rsend)
st_add_test(self_send RANKS 1)
st_add_test(two_queue ARGS 10 64)
