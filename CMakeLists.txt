cmake_minimum_required (VERSION 3.21 FATAL_ERROR)

# set project
project(stream-triggering VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(BUILD_SHARED_LIBS ON)
set(USE_IMPLEMENTATION NONE CACHE STRING "Pick the underlying implementation to use for stream triggering")
set_property(CACHE USE_IMPLEMENTATION PROPERTY STRINGS NONE CUDA MPICH HPE)

if(USE_IMPLEMENTATION STREQUAL NONE)
    message(STATUS "Using a CPU thread as the underlying implementation.")
    add_definitions( -DUSE_THREADS )
elseif(USE_IMPLEMENTATION STREQUAL CUDA)
    message(STATUS "Using CUDA as the underlying implementation")
    set(NEED_CUDA ON)
    add_definitions( -DUSE_CUDA )
elseif(USE_IMPLEMENTATION STREQUAL MPICH)
    message(STATUS "Using MPICH's stream triggering as the underlying implementation.")
elseif(USE_IMPLEMENTATION STREQUAL HPE)
    message(STATUS "Using HPE's stream triggering as the underlying implementation.")
    #enable_language( HIP )
    set(NEED_HIP ON)
    add_definitions( -DUSE_HPE )
else()
    message(FATAL_ERROR "Invalid implementation selected.")
endif()

# Make main library
add_library(stream-triggering "")
set_target_properties(stream-triggering PROPERTIES PUBLIC_HEADER include/stream-triggering.h)

# Deal with extra libraries:
find_package(MPI REQUIRED)
target_link_libraries(stream-triggering MPI::MPI_CXX)

if(NEED_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    target_link_libraries(stream-triggering CUDA::cudart)
    target_link_libraries(stream-triggering CUDA::cuda_driver)
endif()

if(NEED_HIP)
    find_package(hip REQUIRED)
    target_link_libraries(stream-triggering hip::host)
endif()

add_subdirectory(include)
add_subdirectory(source)

install(
	TARGETS stream-triggering
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	PUBLIC_HEADER DESTINATION include
)