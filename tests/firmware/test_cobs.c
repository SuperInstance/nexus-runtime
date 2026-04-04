/**
 * @file test_cobs.c
 * @brief NEXUS COBS encode/decode tests — comprehensive suite.
 *
 * Tests match the reference COBS implementation behavior:
 * - Empty input → [0x01], length 1
 * - Single zero → [0x01, 0x01], length 2
 * - N non-zero → code byte + N data bytes, with code resets at 254
 */

#include "unity.h"
#include "cobs.h"
#include <string.h>

/* Simple LCG PRNG for reproducible random tests */
static uint32_t _seed = 12345;
static uint8_t _rand_byte(void) {
    _seed = _seed * 1103515245 + 12345;
    return (uint8_t)(_seed >> 16);
}

/* ---- Test 1: Basic round-trip with zeros ---- */
TEST_CASE(cobs_roundtrip_with_zeros) {
    uint8_t src[] = {0x01, 0x02, 0x00, 0x03, 0x04, 0x05, 0x00, 0x06};
    uint8_t encoded[32] = {0};
    uint8_t decoded[32] = {0};

    size_t enc_len = cobs_encode(src, sizeof(src), encoded, sizeof(encoded));
    TEST_ASSERT_TRUE(enc_len > 0);

    size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
    TEST_ASSERT_EQUAL_INT(sizeof(src), dec_len);

    for (size_t i = 0; i < sizeof(src); i++) {
        TEST_ASSERT_EQUAL_INT(src[i], decoded[i]);
    }
}

/* ---- Test 2: All zeros ---- */
TEST_CASE(cobs_all_zeros) {
    uint8_t src[] = {0x00, 0x00, 0x00};
    uint8_t encoded[16] = {0};
    uint8_t decoded[16] = {0};

    size_t enc_len = cobs_encode(src, sizeof(src), encoded, sizeof(encoded));
    TEST_ASSERT_TRUE(enc_len > 0);

    /* All zeros: [0x00] → [0x01, 0x01] per zero group + trailing [0x01] */
    /* [0x00,0x00,0x00] → [0x01, 0x01, 0x01, 0x01] */
    TEST_ASSERT_EQUAL_INT(4, (int)enc_len);

    size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
    TEST_ASSERT_EQUAL_INT(sizeof(src), dec_len);

    for (size_t i = 0; i < sizeof(src); i++) {
        TEST_ASSERT_EQUAL_INT(0x00, decoded[i]);
    }
}

/* ---- Test 3: All 0xFF ---- */
TEST_CASE(cobs_all_0xff) {
    uint8_t src[] = {0xFF, 0xFF, 0xFF, 0xFF};
    uint8_t encoded[16] = {0};
    uint8_t decoded[16] = {0};

    size_t enc_len = cobs_encode(src, sizeof(src), encoded, sizeof(encoded));
    TEST_ASSERT_TRUE(enc_len > 0);

    /* 4 non-zero bytes: [0x05, 0xFF, 0xFF, 0xFF, 0xFF] */
    TEST_ASSERT_EQUAL_INT(5, (int)enc_len);
    TEST_ASSERT_EQUAL_INT(0x05, encoded[0]);

    size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
    TEST_ASSERT_EQUAL_INT(sizeof(src), dec_len);

    for (size_t i = 0; i < sizeof(src); i++) {
        TEST_ASSERT_EQUAL_INT(0xFF, decoded[i]);
    }
}

/* ---- Test 4: Empty input ---- */
TEST_CASE(cobs_empty_input) {
    uint8_t buf[4] = {0};
    /* Reference implementation: empty input returns 1 byte [0x01] */
    size_t enc_len = cobs_encode(buf, 0, buf, sizeof(buf));
    TEST_ASSERT_EQUAL_INT(1, (int)enc_len);
    TEST_ASSERT_EQUAL_INT(0x01, buf[0]);

    /* Decode the single-byte encoding */
    uint8_t dec[4] = {0};
    size_t dec_len = cobs_decode(buf, enc_len, dec, sizeof(dec));
    TEST_ASSERT_EQUAL_INT(0, (int)dec_len);
}

/* ---- Test 5: Single byte 0x00 ---- */
TEST_CASE(cobs_single_zero) {
    uint8_t src[] = {0x00};
    uint8_t encoded[8] = {0};
    uint8_t decoded[8] = {0};

    size_t enc_len = cobs_encode(src, 1, encoded, sizeof(encoded));
    /* [0x00] → [0x01, 0x01] */
    TEST_ASSERT_EQUAL_INT(2, (int)enc_len);
    TEST_ASSERT_EQUAL_INT(0x01, encoded[0]);
    TEST_ASSERT_EQUAL_INT(0x01, encoded[1]);

    size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
    TEST_ASSERT_EQUAL_INT(1, (int)dec_len);
    TEST_ASSERT_EQUAL_INT(0x00, decoded[0]);
}

/* ---- Test 6: Single byte 0x01 ---- */
TEST_CASE(cobs_single_nonzero) {
    uint8_t src[] = {0x01};
    uint8_t encoded[8] = {0};
    uint8_t decoded[8] = {0};

    size_t enc_len = cobs_encode(src, 1, encoded, sizeof(encoded));
    TEST_ASSERT_EQUAL_INT(2, (int)enc_len);
    TEST_ASSERT_EQUAL_INT(0x02, encoded[0]);
    TEST_ASSERT_EQUAL_INT(0x01, encoded[1]);

    size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
    TEST_ASSERT_EQUAL_INT(1, (int)dec_len);
    TEST_ASSERT_EQUAL_INT(0x01, decoded[0]);
}

/* ---- Test 7: 254 non-zero bytes ---- */
TEST_CASE(cobs_254_nonzero) {
    uint8_t src[254];
    uint8_t encoded[260] = {0};
    uint8_t decoded[254] = {0};

    for (int i = 0; i < 254; i++) src[i] = (uint8_t)(i + 1);

    size_t enc_len = cobs_encode(src, 254, encoded, sizeof(encoded));
    /* 254 non-zero bytes: first code fills to 0xFF at byte 254, triggers reset
       Result: [0xFF, 254 data bytes, 0x01] = 256 bytes */
    TEST_ASSERT_EQUAL_INT(256, (int)enc_len);
    TEST_ASSERT_EQUAL_INT(0xFF, encoded[0]);

    size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
    TEST_ASSERT_EQUAL_INT(254, (int)dec_len);
    TEST_ASSERT_TRUE(memcmp(src, decoded, 254) == 0);
}

/* ---- Test 8: 255 non-zero bytes ---- */
TEST_CASE(cobs_255_nonzero) {
    uint8_t src[255];
    uint8_t encoded[262] = {0};
    uint8_t decoded[255] = {0};

    for (int i = 0; i < 255; i++) src[i] = (uint8_t)(i + 1);

    size_t enc_len = cobs_encode(src, 255, encoded, sizeof(encoded));
    /* 254 fill first code (0xFF), 255th starts new group
       Result: [0xFF, 254 data, 0x02, 1 data] = 257 bytes */
    TEST_ASSERT_EQUAL_INT(257, (int)enc_len);
    TEST_ASSERT_EQUAL_INT(0xFF, encoded[0]);

    size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
    TEST_ASSERT_EQUAL_INT(255, (int)dec_len);
    TEST_ASSERT_TRUE(memcmp(src, decoded, 255) == 0);
}

/* ---- Test 9: 100 random round-trips (reduced from 1000 for speed) ---- */
TEST_CASE(cobs_100_random_roundtrips) {
    uint8_t src[256];
    uint8_t encoded[300];
    uint8_t decoded[256];
    _seed = 42;

    for (int trial = 0; trial < 1000; trial++) {
        size_t len = (_rand_byte() % 200) + 1;
        for (size_t i = 0; i < len; i++) {
            src[i] = _rand_byte();
        }

        size_t enc_len = cobs_encode(src, len, encoded, sizeof(encoded));
        TEST_ASSERT_TRUE(enc_len > 0);

        size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
        TEST_ASSERT_EQUAL_INT((int)len, (int)dec_len);

        TEST_ASSERT_TRUE(memcmp(src, decoded, len) == 0);
    }
}

/* ---- Test 10: Large random buffer (1000 bytes) ---- */
TEST_CASE(cobs_large_random_1000) {
    uint8_t src[1000];
    uint8_t encoded[1100];
    uint8_t decoded[1000];
    _seed = 999;

    for (int i = 0; i < 1000; i++) src[i] = _rand_byte();

    size_t enc_len = cobs_encode(src, 1000, encoded, sizeof(encoded));
    TEST_ASSERT_TRUE(enc_len > 1000);
    TEST_ASSERT_TRUE(enc_len <= 1008);

    size_t dec_len = cobs_decode(encoded, enc_len, decoded, sizeof(decoded));
    TEST_ASSERT_EQUAL_INT(1000, (int)dec_len);
    TEST_ASSERT_TRUE(memcmp(src, decoded, 1000) == 0);
}

/* ---- Test 11: NULL inputs ---- */
TEST_CASE(cobs_null_inputs) {
    uint8_t buf[32] = {0};
    TEST_ASSERT_EQUAL_INT(0, (int)cobs_encode(NULL, 10, buf, sizeof(buf)));
    TEST_ASSERT_EQUAL_INT(0, (int)cobs_encode(buf, 10, NULL, 32));
    TEST_ASSERT_EQUAL_INT(0, (int)cobs_decode(NULL, 10, buf, sizeof(buf)));
    TEST_ASSERT_EQUAL_INT(0, (int)cobs_decode(buf, 10, NULL, 32));
}

/* ---- Test 12: Buffer too small ---- */
TEST_CASE(cobs_buffer_too_small) {
    uint8_t src[] = {0x01, 0x02, 0x03};
    uint8_t encoded[2] = {0};

    size_t enc_len = cobs_encode(src, sizeof(src), encoded, sizeof(encoded));
    TEST_ASSERT_EQUAL_INT(0, (int)enc_len);
}
