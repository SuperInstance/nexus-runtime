# NEXUS Firmware Tests - Unity test framework runner

cmake_minimum_required(VERSION 3.16)
project(nexus_firmware_tests C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter")

# Include paths for shared headers and source components
include_directories(
    ${CMAKE_SOURCE_DIR}/shared
    ${CMAKE_SOURCE_DIR}/firmware/nexus_vm/include
    ${CMAKE_SOURCE_DIR}/firmware/nexus_vm
    ${CMAKE_SOURCE_DIR}/firmware/wire_protocol
    ${CMAKE_SOURCE_DIR}/firmware/safety
    ${CMAKE_SOURCE_DIR}/firmware/drivers
    ${CMAKE_SOURCE_DIR}/tests/firmware
)

# Test sources
add_executable(test_firmware
    test_main.c
    test_vm_opcodes.c
    test_cobs.c
    test_crc16.c
    test_frame.c
    # Include source files under test
    ${CMAKE_SOURCE_DIR}/firmware/wire_protocol/cobs.c
    ${CMAKE_SOURCE_DIR}/firmware/wire_protocol/crc16.c
    ${CMAKE_SOURCE_DIR}/firmware/wire_protocol/frame.c
    ${CMAKE_SOURCE_DIR}/firmware/wire_protocol/message.c
    ${CMAKE_SOURCE_DIR}/firmware/nexus_vm/vm_core.c
    ${CMAKE_SOURCE_DIR}/firmware/nexus_vm/vm_opcodes.c
    ${CMAKE_SOURCE_DIR}/firmware/nexus_vm/vm_validate.c
)

enable_testing()
add_test(NAME firmware_tests COMMAND test_firmware)
