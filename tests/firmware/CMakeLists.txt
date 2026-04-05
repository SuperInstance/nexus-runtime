# NEXUS Firmware Tests - Unity test framework runner

cmake_minimum_required(VERSION 3.16)
project(nexus_firmware_tests C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter")

# Include paths for shared headers and source components
include_directories(
    ${CMAKE_SOURCE_DIR}/shared
    ${CMAKE_SOURCE_DIR}/firmware/src/core/nexus_vm/include
    ${CMAKE_SOURCE_DIR}/firmware/src/core/nexus_vm
    ${CMAKE_SOURCE_DIR}/firmware/src/core/wire_protocol
    ${CMAKE_SOURCE_DIR}/firmware/src/safety
    ${CMAKE_SOURCE_DIR}/firmware/src/drivers
    ${CMAKE_SOURCE_DIR}/tests/firmware
)

# Test sources
add_executable(test_firmware
    test_main.c
    test_vm_opcodes.c
    test_cobs.c
    test_crc16.c
    test_frame.c
    test_hal_integration.c
    # Include source files under test
    ${CMAKE_SOURCE_DIR}/firmware/src/core/wire_protocol/cobs.c
    ${CMAKE_SOURCE_DIR}/firmware/src/core/wire_protocol/crc16.c
    ${CMAKE_SOURCE_DIR}/firmware/src/core/wire_protocol/frame.c
    ${CMAKE_SOURCE_DIR}/firmware/src/core/wire_protocol/message.c
    ${CMAKE_SOURCE_DIR}/firmware/src/core/nexus_vm/vm_core.c
    ${CMAKE_SOURCE_DIR}/firmware/src/core/nexus_vm/vm_opcodes.c
    ${CMAKE_SOURCE_DIR}/firmware/src/core/nexus_vm/vm_validate.c
    ${CMAKE_SOURCE_DIR}/firmware/src/safety/safety_sm.c
    ${CMAKE_SOURCE_DIR}/firmware/src/safety/watchdog.c
    ${CMAKE_SOURCE_DIR}/firmware/src/safety/heartbeat.c
    ${CMAKE_SOURCE_DIR}/firmware/src/drivers/hal.c
)

target_link_libraries(test_firmware m)

enable_testing()
add_test(NAME firmware_tests COMMAND test_firmware)
