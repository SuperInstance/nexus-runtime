/**
 * @file test_crc16.c
 * @brief NEXUS CRC-16/CCITT-FALSE tests — comprehensive suite.
 */

#include "unity.h"
#include "crc16.h"
#include <string.h>

/* ---- Test 1: Check value 0x29B1 for "123456789" ---- */
TEST_CASE(crc16_check_value) {
    const uint8_t *data = (const uint8_t *)"123456789";
    uint16_t crc = crc16_ccitt(data, 9);
    TEST_ASSERT_EQUAL_INT(0x29B1, crc);
}

/* ---- Test 2: Empty input returns initial value 0xFFFF ---- */
TEST_CASE(crc16_empty) {
    uint16_t crc = crc16_ccitt(NULL, 0);
    TEST_ASSERT_EQUAL_INT(0xFFFF, crc);
}

/* ---- Test 3: Single byte zero ---- */
TEST_CASE(crc16_single_byte_zero) {
    uint8_t data[] = {0x00};
    uint16_t crc = crc16_ccitt(data, 1);
    /* Manually verified: CRC-16/CCITT-FALSE of [0x00] = 0xE1F0 */
    TEST_ASSERT_EQUAL_INT(0xE1F0, crc);
}

/* ---- Test 4: All zeros ---- */
TEST_CASE(crc16_all_zeros) {
    uint8_t data[256];
    memset(data, 0x00, sizeof(data));
    uint16_t crc = crc16_ccitt(data, 256);
    TEST_ASSERT_TRUE(crc != 0xFFFF);
    TEST_ASSERT_TRUE(crc != 0x0000);
}

/* ---- Test 5: Large buffer ---- */
TEST_CASE(crc16_large_buffer) {
    uint8_t data[1000];
    for (int i = 0; i < 1000; i++) data[i] = (uint8_t)i;
    uint16_t crc = crc16_ccitt(data, 1000);
    TEST_ASSERT_TRUE(crc != 0x0000);
}

/* ---- Test 6: All 0xFF ---- */
TEST_CASE(crc16_all_ff) {
    uint8_t data[10];
    memset(data, 0xFF, sizeof(data));
    uint16_t crc = crc16_ccitt(data, 10);
    TEST_ASSERT_TRUE(crc != 0x0000);
    TEST_ASSERT_TRUE(crc != 0xFFFF);
}

/* ---- Test 7: Consistency check ---- */
TEST_CASE(crc16_consistency) {
    uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09};
    uint16_t crc1 = crc16_ccitt(data, 9);
    uint16_t crc2 = crc16_ccitt(data, 9);
    TEST_ASSERT_EQUAL_INT(crc1, crc2);
}
