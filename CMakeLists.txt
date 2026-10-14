cmake_minimum_required (VERSION 3.24 FATAL_ERROR)

# Version
set(CURR_VERSION 0.8.0)

# Set CMake Build Type
 if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "No CMAKE_BUILD_TYPE specified, setting build type to RELEASE.")
        set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "Build type for project." FORCE)
 endif()

# Backend settings
option(USE_THREAD_BACKEND "Enable the THREAD backend for the queue objects" ON)
option(USE_HIP_BACKEND "Enable the HIP backend for the queue objects" OFF)
option(USE_CUDA_BACKEND "Enable the CUDA backend for the queue objects" OFF)
option(USE_CXI_BACKEND "Enable the CXI backend for the queue objects" OFF)
option(BUILD_TESTS "Build the multi-backend tests (run them with ctest)" OFF)

if(USE_CUDA_BACKEND)
    project(stream-triggering VERSION ${CURR_VERSION} LANGUAGES CUDA C CXX)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES native)
elseif(USE_HIP_BACKEND)
    project(stream-triggering VERSION ${CURR_VERSION} LANGUAGES HIP C CXX)
else() 
    project(stream-triggering VERSION ${CURR_VERSION} LANGUAGES C CXX)
endif()

# Language settings
set(CMAKE_CXX_STANDARD 23)
set(BUILD_SHARED_LIBS ON)

# Make main library
add_library(stream-triggering "")
set_target_properties(stream-triggering PROPERTIES PUBLIC_HEADER include/stream-triggering.h)

### Third Party Library Settings

# Update the path that CMAKE will search for Find_XXX.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find MPI:
find_package(MPI REQUIRED)
target_link_libraries(stream-triggering PUBLIC MPI::MPI_CXX)

# Add definitions (and extra settings) for the chosen backend
if(USE_THREAD_BACKEND)
    message(STATUS "Enabling the THREAD based backend")
    add_definitions( -DUSE_THREADS )
endif()

if(USE_HIP_BACKEND)
    message(STATUS "Enabling the HIP based backend")
    set(GPU_TYPE "AMD")
    add_definitions( -DUSE_MEM_OPS )
endif()

if(USE_CUDA_BACKEND)
    message(STATUS "Enabling the CUDA based backend")
    set(GPU_TYPE "NVIDIA")
    add_definitions( -DUSE_MEM_OPS )
endif()

if(USE_CXI_BACKEND)
    message(STATUS "Enabling HPE's CXI Libfabric provider based backend")
    set(GPU_TYPE "AMD")
    add_definitions( -DUSE_CXI )
    find_package(LibFabric REQUIRED)
    target_include_directories(stream-triggering PUBLIC ${LIBFABRIC_INCLUDE_DIR})
    target_link_libraries(stream-triggering PRIVATE ${LIBFABRIC_LIBRARY})
    if("${CMAKE_HIP_ARCHITECTURES}" STREQUAL "gfx90a")
        message(STATUS "Adding custom code for the gfx90a")
        add_definitions(-DUSE_TIOGA)
    elseif("${CMAKE_HIP_ARCHITECTURES}" STREQUAL "gfx942")
        message(STATUS "Adding custom code for the gfx942")
        add_definitions(-DUSE_TUOLUMNE)
    endif()
endif()

# Link with the GPU runtime the chosen backends need
if(GPU_TYPE STREQUAL AMD)
    message(STATUS "Linking with HIP")
    find_package(hip REQUIRED)
    target_link_libraries(stream-triggering PRIVATE hip::host)
    add_definitions( -DHIP_GPUS )
elseif(GPU_TYPE STREQUAL NVIDIA)
    message(STATUS "Linking with CUDA")
    find_package(CUDAToolkit)
    target_link_libraries(stream-triggering PRIVATE CUDA::cudart CUDA::cuda_driver)
    add_definitions( -DCUDA_GPUS )
endif()

add_subdirectory(include)
add_subdirectory(source)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS stream-triggering
	    EXPORT stream-triggeringTargets)

##### Code for CMake Packaging
include(CMakePackageConfigHelpers)

# Create the configuration file
configure_package_config_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/stream-triggering-config.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
	INSTALL_DESTINATION share/${PROJECT_NAME}
)

## Version file
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config-version.cmake
	COMPATIBILITY SameMinorVersion
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config-version.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
	DESTINATION share/${PROJECT_NAME}
)

# Install everything
install(EXPORT stream-triggeringTargets
	NAMESPACE MPIAdvance::
	DESTINATION share/${PROJECT_NAME}
)
