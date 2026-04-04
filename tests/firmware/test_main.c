/**
 * @file test_main.c
 * @brief NEXUS firmware test runner - Unity framework.
 */

#include "unity.h"
#include <stdlib.h>

/* External test functions */
extern void test_vm_opcodes_nop(void);
extern void test_vm_load_bytecode(void);
extern void test_vm_validate_valid(void);
extern void test_vm_validate_invalid_size(void);
extern void test_cobs_encode_decode(void);
extern void test_cobs_empty(void);
extern void test_crc16_check_value(void);
extern void test_crc16_empty(void);
extern void test_frame_init(void);
extern void test_frame_feed_null(void);

void setUp(void) {
    /* Called before each test */
}

void tearDown(void) {
    /* Called after each test */
}

int main(void) {
    printf("=== NEXUS Firmware Tests ===\n\n");

    (void)UNITY_BEGIN();

    /* VM opcode tests */
    RUN_TEST(vm_opcodes_nop);
    RUN_TEST(vm_load_bytecode);
    RUN_TEST(vm_validate_valid);
    RUN_TEST(vm_validate_invalid_size);

    /* Wire protocol tests */
    RUN_TEST(cobs_encode_decode);
    RUN_TEST(cobs_empty);
    RUN_TEST(crc16_check_value);
    RUN_TEST(crc16_empty);
    RUN_TEST(frame_init);
    RUN_TEST(frame_feed_null);

    return UNITY_END();
}
