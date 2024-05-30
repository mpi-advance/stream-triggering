cmake_minimum_required (VERSION 3.17 FATAL_ERROR)

# set project
project(stream-triggering VERSION 0.1 LANGUAGES C CXX)

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

# Make main library
add_library(stream-triggering "")

add_subdirectory(include)
add_subdirectory(source)