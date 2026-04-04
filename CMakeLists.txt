# NEXUS Runtime - Top-level monorepo CMake configuration
#
# This CMakeLists.txt provides a host build target for testing.
# For ESP32 firmware builds, use the firmware/ directory with ESP-IDF.

cmake_minimum_required(VERSION 3.16)
project(nexus_runtime C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter")

# Only build tests if not using ESP-IDF (host build)
if(NOT DEFINED ENV{IDF_PATH})
    message(STATUS "Host build detected - enabling test targets")
    add_subdirectory(tests/firmware)
endif()
