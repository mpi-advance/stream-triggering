cmake_minimum_required (VERSION 3.17 FATAL_ERROR)

# set project
project(stream-triggering VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(BUILD_SHARED_LIBS ON)
set(USE_IMPLEMENTATION NONE CACHE STRING "Pick the underlying implementation to use for stream triggering")
set_property(CACHE USE_IMPLEMENTATION PROPERTY STRINGS NONE MPICH HPE)

if(USE_IMPLEMENTATION STREQUAL NONE)
    message(STATUS "Using no underlying implementation.")
elseif(USE_IMPLEMENTATION STREQUAL MPICH)
    message(STATUS "Using HPE's stream triggering as the underlying implementation.")
elseif(USE_IMPLEMENTATION STREQUAL HPE)
    message(STATUS "Using MPICH's stream triggering as the underlying implementation.")
else()
    message(FATAL_ERROR "Invalid implementation selected.")
endif()

find_package(MPI REQUIRED)

# Make main library
add_library(stream-triggering "")
target_link_libraries(stream-triggering MPI::MPI_C)

set_target_properties(stream-triggering PROPERTIES PUBLIC_HEADER include/stream-triggering.h)

add_subdirectory(include)
add_subdirectory(source)

install(
	TARGETS stream-triggering
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	PUBLIC_HEADER DESTINATION include
)