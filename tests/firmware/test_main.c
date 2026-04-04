/**
 * @file test_main.c
 * @brief NEXUS firmware test runner - Unity framework.
 *
 * Runs: COBS, CRC-16, Frame, and HAL integration tests.
 * VM opcode tests are in test_vm_opcodes.c with their own main().
 */

#include "unity.h"
#include <stdio.h>
#include <stdlib.h>

/* Wire protocol tests - COBS */
extern void test_cobs_roundtrip_with_zeros(void);
extern void test_cobs_all_zeros(void);
extern void test_cobs_all_0xff(void);
extern void test_cobs_empty_input(void);
extern void test_cobs_single_zero(void);
extern void test_cobs_single_nonzero(void);
extern void test_cobs_254_nonzero(void);
extern void test_cobs_255_nonzero(void);
extern void test_cobs_100_random_roundtrips(void);
extern void test_cobs_large_random_1000(void);
extern void test_cobs_null_inputs(void);
extern void test_cobs_buffer_too_small(void);

/* Wire protocol tests - CRC-16 */
extern void test_crc16_check_value(void);
extern void test_crc16_empty(void);
extern void test_crc16_single_byte_zero(void);
extern void test_crc16_all_zeros(void);
extern void test_crc16_large_buffer(void);
extern void test_crc16_all_ff(void);
extern void test_crc16_consistency(void);

/* Wire protocol tests - Frame */
extern void test_frame_init(void);
extern void test_frame_feed_null(void);
extern void test_frame_heartbeat_roundtrip(void);
extern void test_frame_device_identity_with_payload(void);
extern void test_frame_reflex_deploy_binary(void);
extern void test_frame_crc_mismatch(void);
extern void test_frame_oversized_rejection(void);
extern void test_frame_too_short(void);
extern void test_frame_multiple_sequence(void);
extern void test_frame_tx_build_valid(void);
extern void test_frame_tx_build_with_payload(void);

/* HAL integration tests */
extern void test_hal_test_01_vm_reads_sensor(void);
extern void test_hal_test_02_vm_writes_actuator(void);
extern void test_hal_test_03_rate_limiting(void);
extern void test_hal_test_04_estop_forces_safe(void);
extern void test_hal_test_05_nan_guard(void);
extern void test_hal_test_06_safety_state_transitions(void);
extern void test_hal_test_07_watchdog_timeout(void);
extern void test_hal_test_08_overcurrent(void);
extern void test_hal_test_09_clamp_write_integration(void);
extern void test_hal_test_10_actuators_independent(void);
extern void test_hal_test_init_defaults(void);
extern void test_hal_test_configure_sensor(void);
extern void test_hal_test_read_sensor_invalid(void);
extern void test_hal_test_infinity_guard(void);
extern void test_hal_test_clamp_actuator(void);
extern void test_hal_test_watchdog_via_hal(void);
extern void test_hal_test_safety_state_name(void);
extern void test_hal_test_safety_sm_basic(void);
extern void test_hal_test_heartbeat_monitor(void);
extern void test_hal_test_rate_negative_direction(void);

void setUp(void) { }
void tearDown(void) { }

int main(void) {
    printf("=== NEXUS Firmware Tests ===\n\n");

    /* COBS tests */
    printf("--- COBS Encode/Decode Tests ---\n");
    RUN_TEST(cobs_roundtrip_with_zeros);
    RUN_TEST(cobs_all_zeros);
    RUN_TEST(cobs_all_0xff);
    RUN_TEST(cobs_empty_input);
    RUN_TEST(cobs_single_zero);
    RUN_TEST(cobs_single_nonzero);
    RUN_TEST(cobs_254_nonzero);
    RUN_TEST(cobs_255_nonzero);
    RUN_TEST(cobs_100_random_roundtrips);
    RUN_TEST(cobs_large_random_1000);
    RUN_TEST(cobs_null_inputs);
    RUN_TEST(cobs_buffer_too_small);

    /* CRC-16 tests */
    printf("\n--- CRC-16/CCITT-FALSE Tests ---\n");
    RUN_TEST(crc16_check_value);
    RUN_TEST(crc16_empty);
    RUN_TEST(crc16_single_byte_zero);
    RUN_TEST(crc16_all_zeros);
    RUN_TEST(crc16_large_buffer);
    RUN_TEST(crc16_all_ff);
    RUN_TEST(crc16_consistency);

    /* Frame tests */
    printf("\n--- Frame Reception Tests ---\n");
    RUN_TEST(frame_init);
    RUN_TEST(frame_feed_null);
    RUN_TEST(frame_heartbeat_roundtrip);
    RUN_TEST(frame_device_identity_with_payload);
    RUN_TEST(frame_reflex_deploy_binary);
    RUN_TEST(frame_crc_mismatch);
    RUN_TEST(frame_oversized_rejection);
    RUN_TEST(frame_too_short);
    RUN_TEST(frame_multiple_sequence);
    RUN_TEST(frame_tx_build_valid);
    RUN_TEST(frame_tx_build_with_payload);

    /* HAL integration tests */
    printf("\n--- HAL Integration Tests ---\n");
    RUN_TEST(hal_test_init_defaults);
    RUN_TEST(hal_test_configure_sensor);
    RUN_TEST(hal_test_read_sensor_invalid);
    RUN_TEST(hal_test_01_vm_reads_sensor);
    RUN_TEST(hal_test_02_vm_writes_actuator);
    RUN_TEST(hal_test_03_rate_limiting);
    RUN_TEST(hal_test_04_estop_forces_safe);
    RUN_TEST(hal_test_05_nan_guard);
    RUN_TEST(hal_test_infinity_guard);
    RUN_TEST(hal_test_clamp_actuator);
    RUN_TEST(hal_test_rate_negative_direction);
    RUN_TEST(hal_test_06_safety_state_transitions);
    RUN_TEST(hal_test_safety_state_name);
    RUN_TEST(hal_test_safety_sm_basic);
    RUN_TEST(hal_test_07_watchdog_timeout);
    RUN_TEST(hal_test_watchdog_via_hal);
    RUN_TEST(hal_test_heartbeat_monitor);
    RUN_TEST(hal_test_08_overcurrent);
    RUN_TEST(hal_test_09_clamp_write_integration);
    RUN_TEST(hal_test_10_actuators_independent);

    printf("\n=== All tests complete ===\n");
    return 0;
}
