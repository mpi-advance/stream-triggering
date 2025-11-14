cmake_minimum_required (VERSION 3.24 FATAL_ERROR)

# set project
project(stream-triggering VERSION 0.5.0 LANGUAGES C CXX)

# Backend settings
option(USE_THREAD_BACKEND "Enable the THREAD backend for the queue objects" ON)
option(USE_HIP_BACKEND "Enable the HIP backend for the queue objects" OFF)
option(USE_CUDA_BACKEND "Enable the CUDA backend for the queue objects" OFF)
option(USE_CXI_BACKEND "Enable the CXI backend for the queue objects" OFF)

# Language settings
set(CMAKE_CXX_STANDARD 23)
set(BUILD_SHARED_LIBS ON)

if(USE_CUDA_BACKEND)
    # This must come before making any targets or else cmake falls apart!
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Make main library
add_library(stream-triggering "")
set_target_properties(stream-triggering PROPERTIES PUBLIC_HEADER include/stream-triggering.h)

### Third Party Library Settings

# Update the path that CMAKE will search for Find_XXX.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find MPI:
find_package(MPI REQUIRED)
target_link_libraries(stream-triggering MPI::MPI_CXX)

# Link with NVIDIA stuff
function(add_cuda target_name)
    message(STATUS "Linking with CUDA")
    find_package(CUDAToolkit)
    target_link_libraries(${target_name} CUDA::cudart)
    target_link_libraries(${target_name} CUDA::cuda_driver)
    add_definitions( -DCUDA_GPUS )
endfunction()

# Link with AMD Stuff
function(add_hip target_name)
    message(STATUS "Linking with HIP")
    find_package(hip REQUIRED)
    target_link_libraries(${target_name} hip::host)
    add_definitions( -DHIP_GPUS )
endfunction()
###

# Add definitions (and extra settings) for the chosen backend
if(USE_THREAD_BACKEND)
    message(STATUS "Enabling the THREAD based backend")
    add_definitions( -DUSE_THREADS )
endif()

if(USE_HIP_BACKEND)
    message(STATUS "Enabling the HIP based backend")
    set(NEED_GPU ON)
    set(GPU_TYPE "AMD")
    add_definitions( -DUSE_MEM_OPS )
endif()

if(USE_CUDA_BACKEND)
    message(STATUS "Enabling the CUDA based backend")
    set(NEED_GPU ON)
    set(GPU_TYPE "NVIDIA")
    add_definitions( -DUSE_MEM_OPS )
endif()

if(USE_CXI_BACKEND)
    message(STATUS "Enabling HPE's CXI Libfabric provider based backend")
    set(NEED_GPU ON)
    set(GPU_TYPE "AMD")
    add_definitions( -DUSE_CXI )
    find_package(LibFabric REQUIRED)
    target_include_directories(stream-triggering PUBLIC ${LIBFABRIC_INCLUDE_DIR})
    target_link_libraries(stream-triggering ${LIBFABRIC_LIBRARY})
    target_compile_options(stream-triggering PUBLIC -x hip)
    if("${CMAKE_HIP_ARCHITECTURES}" STREQUAL "gfx90a")
        message(STATUS "Adding custom code for the gfx90a")
        add_definitions(-DUSE_TIOGA)
    elseif("${CMAKE_HIP_ARCHITECTURES}" STREQUAL "gfx942")
        message(STATUS "Adding custom code for the gfx942")
        add_definitions(-DUSE_TUOLUMNE)
    endif()
endif()

if( (CMAKE_CXX_COMPILER_ID STREQUAL GNU AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13) OR
    (CMAKE_CXX_COMPILER_ID STREQUAL Clang AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 17))
    message(STATUS "Using advanced C++ features")
    add_definitions( -DADVANCED_CPP23 )
endif()

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
