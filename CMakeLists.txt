cmake_minimum_required (VERSION 3.24 FATAL_ERROR)

# set project
project(stream-triggering VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(BUILD_SHARED_LIBS ON)
set(USE_IMPLEMENTATION NONE CACHE STRING "Pick the underlying implementation to use for stream triggering")
set_property(CACHE USE_IMPLEMENTATION PROPERTY STRINGS NONE GPU_MEM_OPS MPICH HPE CXI)
set_property(CACHE GPU_TYPE PROPERTY STRINGS AMD NVIDIA)

# Update the path that CMAKE will search for Find_XXX.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Make main library
add_library(stream-triggering "")
set_target_properties(stream-triggering PROPERTIES PUBLIC_HEADER include/stream-triggering.h)

# Deal with extra libraries:
find_package(MPI REQUIRED)
target_link_libraries(stream-triggering MPI::MPI_CXX)

# Handle extra requirments based on backend
if(USE_IMPLEMENTATION STREQUAL NONE)
    message(STATUS "Using a CPU thread as the underlying implementation.")
    add_definitions( -DUSE_THREADS )
elseif(USE_IMPLEMENTATION STREQUAL GPU_MEM_OPS)
    message(STATUS "Using GPU_MEM_OPS as the underlying implementation")
    set(NEED_GPU ON)
    add_definitions( -DUSE_MEM_OPS )
elseif(USE_IMPLEMENTATION STREQUAL MPICH)
    message(STATUS "Using MPICH's stream triggering as the underlying implementation.")
elseif(USE_IMPLEMENTATION STREQUAL HPE)
    message(STATUS "Using HPE's stream triggering as the underlying implementation.")
    set(NEED_GPU ON)
    set(GPU_TYPE "AMD")
    add_definitions( -DUSE_HPE )
elseif(USE_IMPLEMENTATION STREQUAL CXI)
    message(STATUS "Using HPE's CXI Libfabric provider as the underlying implementation.")
    set(NEED_GPU ON)
    set(GPU_TYPE "AMD")
    add_definitions( -DUSE_CXI )
    find_package(LibFabric REQUIRED)
    target_include_directories(stream-triggering PUBLIC ${LIBFABRIC_INCLUDE_DIR})
    target_link_libraries(stream-triggering ${LIBFABRIC_LIBRARY})
else()
    message(FATAL_ERROR "Invalid implementation selected.")
endif()

# Link with NVIDIA stuff
function(add_cuda target_name)
    message(STATUS "Linking with CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit)
    target_link_libraries(${target_name} CUDA::cudart)
    target_link_libraries(${target_name} CUDA::cuda_driver)
    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
    add_definitions( -DCUDA_GPUS )
endfunction()

# Link with AMD Stuff
function(add_hip target_name)
    message(STATUS "Linking with HIP")
    enable_language(HIP)
    find_package(hip REQUIRED)
    target_link_libraries(${target_name} hip::host)
    add_definitions( -DHIP_GPUS )
endfunction()

if(NEED_GPU)
    if(GPU_TYPE STREQUAL AMD)
        add_hip(stream-triggering)
    elseif(GPU_TYPE STREQUAL NVIDIA)
        add_cuda(stream-triggering)
    else()
        message(FATAL_ERROR "Invalid GPU Type selected.")
    endif()
endif()

add_subdirectory(include)
add_subdirectory(source)

install(
	TARGETS stream-triggering
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	PUBLIC_HEADER DESTINATION include
)