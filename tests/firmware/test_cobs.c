/**
 * @file test_cobs.c
 * @brief NEXUS COBS encode/decode tests (stub).
 */

#include "unity.h"
#include "cobs.h"

TEST_CASE(cobs_encode_decode) {
    uint8_t src[] = {0x01, 0x02, 0x00, 0x03, 0x04, 0x05, 0x00, 0x06};
    uint8_t encoded[32] = {0};
    uint8_t decoded[32] = {0};

    size_t enc_len = cobs_encode(src, sizeof(src), encoded, sizeof(encoded));
    TEST_ASSERT_TRUE(enc_len > 0);

    size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
    TEST_ASSERT_TRUE(dec_len == sizeof(src));

    /* Verify round-trip */
    for (size_t i = 0; i < sizeof(src); i++) {
        TEST_ASSERT_EQUAL_INT8(src[i], decoded[i]);
    }
}

TEST_CASE(cobs_empty) {
    uint8_t encoded[4] = {0};

    size_t enc_len = cobs_encode(NULL, 0, encoded, sizeof(encoded));
    TEST_ASSERT_TRUE(enc_len == 0);
}
