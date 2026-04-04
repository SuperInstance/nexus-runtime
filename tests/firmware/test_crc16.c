/**
 * @file test_crc16.c
 * @brief NEXUS CRC-16/CCITT-FALSE tests (stub).
 */

#include "unity.h"
#include "crc16.h"

TEST_CASE(crc16_check_value) {
    /* Check value for "123456789" must be 0x29B1 */
    const uint8_t *data = (const uint8_t *)"123456789";
    uint16_t crc = crc16_ccitt(data, 9);
    TEST_ASSERT_EQUAL_INT(0x29B1, crc);
}

TEST_CASE(crc16_empty) {
    uint16_t crc = crc16_ccitt(NULL, 0);
    TEST_ASSERT_EQUAL_INT(0xFFFF, crc);
}
