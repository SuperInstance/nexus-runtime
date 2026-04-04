# NEXUS Firmware - ESP-IDF Project
# Top-level ESP-IDF CMakeLists.txt for ESP32-S3 target

cmake_minimum_required(VERSION 3.16)

# Include ESP-IDF build system
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(nexus_firmware)

# Register components - ESP-IDF will recurse into these directories
# Main component is always required
# nexus_vm: Bytecode VM interpreter
# wire_protocol: COBS/CRC/frame/dispatch
# safety: Safety state machine, watchdog, heartbeat
# drivers: Sensor bus and actuator drivers
