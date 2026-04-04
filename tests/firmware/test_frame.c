/**
 * @file test_frame.c
 * @brief NEXUS frame reception tests — comprehensive suite.
 */

#include "unity.h"
#include "frame.h"
#include "cobs.h"
#include "crc16.h"
#include "message.h"
#include <string.h>

/* Helper: build a wire frame manually for testing */
static size_t build_wire_frame(uint8_t msg_type, uint8_t flags,
                               uint16_t seq, uint32_t ts,
                               const uint8_t *payload, uint16_t payload_len,
                               uint8_t *out, size_t out_max) {
    uint8_t header[FRAME_HEADER_SIZE];
    header[0] = msg_type;
    header[1] = flags;
    header[2] = (uint8_t)(seq >> 8);
    header[3] = (uint8_t)(seq & 0xFF);
    header[4] = (uint8_t)(ts >> 24);
    header[5] = (uint8_t)(ts >> 16);
    header[6] = (uint8_t)(ts >> 8);
    header[7] = (uint8_t)(ts & 0xFF);
    header[8] = (uint8_t)(payload_len >> 8);
    header[9] = (uint8_t)(payload_len & 0xFF);

    uint8_t data[FRAME_MAX_DECODED];
    size_t data_len = FRAME_HEADER_SIZE + payload_len;
    memcpy(data, header, FRAME_HEADER_SIZE);
    if (payload_len > 0 && payload) {
        memcpy(data + FRAME_HEADER_SIZE, payload, payload_len);
    }

    uint16_t crc = crc16_ccitt(data, data_len);
    data[data_len] = (uint8_t)(crc >> 8);
    data[data_len + 1] = (uint8_t)(crc & 0xFF);
    data_len += 2;

    uint8_t cobs[FRAME_MAX_COBS];
    size_t cobs_len = cobs_encode(data, data_len, cobs, sizeof(cobs));
    if (cobs_len == 0) return 0;

    size_t wire_len = 1 + cobs_len + 1;
    if (wire_len > out_max) return 0;

    out[0] = 0x00;
    memcpy(out + 1, cobs, cobs_len);
    out[1 + cobs_len] = 0x00;
    return wire_len;
}

/* Helper: build a wire frame with intentionally wrong CRC */
static size_t build_bad_crc_frame(uint8_t msg_type, uint8_t flags,
                                  uint16_t seq, uint32_t ts,
                                  const uint8_t *payload, uint16_t payload_len,
                                  uint8_t *out, size_t out_max) {
    uint8_t header[FRAME_HEADER_SIZE];
    header[0] = msg_type;
    header[1] = flags;
    header[2] = (uint8_t)(seq >> 8);
    header[3] = (uint8_t)(seq & 0xFF);
    header[4] = (uint8_t)(ts >> 24);
    header[5] = (uint8_t)(ts >> 16);
    header[6] = (uint8_t)(ts >> 8);
    header[7] = (uint8_t)(ts & 0xFF);
    header[8] = (uint8_t)(payload_len >> 8);
    header[9] = (uint8_t)(payload_len & 0xFF);

    uint8_t data[FRAME_MAX_DECODED];
    size_t data_len = FRAME_HEADER_SIZE + payload_len;
    memcpy(data, header, FRAME_HEADER_SIZE);
    if (payload_len > 0 && payload) {
        memcpy(data + FRAME_HEADER_SIZE, payload, payload_len);
    }

    /* Intentionally wrong CRC */
    data[data_len] = 0xDE;
    data[data_len + 1] = 0xAD;
    data_len += 2;

    uint8_t cobs[FRAME_MAX_COBS];
    size_t cobs_len = cobs_encode(data, data_len, cobs, sizeof(cobs));
    if (cobs_len == 0) return 0;

    size_t wire_len = 1 + cobs_len + 1;
    if (wire_len > out_max) return 0;

    out[0] = 0x00;
    memcpy(out + 1, cobs, cobs_len);
    out[1 + cobs_len] = 0x00;
    return wire_len;
}

/* ---- Test 1: Init state ---- */
TEST_CASE(frame_init) {
    frame_receiver_t rx;
    frame_receiver_init(&rx);
    TEST_ASSERT_EQUAL_INT(FRAME_STATE_IDLE, rx.state);
    TEST_ASSERT_EQUAL_INT(0, rx.cobs_idx);
    TEST_ASSERT_EQUAL_INT(0, rx.error_count);
}

/* ---- Test 2: Feed NULL ---- */
TEST_CASE(frame_feed_null) {
    uint32_t err = frame_receiver_feed(NULL, 0x00);
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_BUFFER_OVERFLOW, err);
}

/* ---- Test 3: Valid HEARTBEAT frame (type 0x05, no payload) ---- */
TEST_CASE(frame_heartbeat_roundtrip) {
    uint8_t wire[FRAME_MAX_WIRE];
    size_t wire_len = build_wire_frame(0x05, 0x00, 1, 1000, NULL, 0,
                                       wire, sizeof(wire));
    TEST_ASSERT_TRUE(wire_len > 0);

    frame_receiver_t rx;
    frame_receiver_init(&rx);

    uint32_t result = FRAME_ERR_NONE;
    for (size_t i = 0; i < wire_len; i++) {
        result = frame_receiver_feed(&rx, wire[i]);
    }
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_NONE, result);

    uint16_t data_len = 0;
    const uint8_t *data = frame_receiver_get_data(&rx, &data_len);
    TEST_ASSERT_NOT_NULL(data);
    TEST_ASSERT_EQUAL_INT(FRAME_HEADER_SIZE + FRAME_CRC_SIZE, data_len);

    /* Verify header */
    TEST_ASSERT_EQUAL_INT(0x05, data[0]);  /* msg_type */
    TEST_ASSERT_EQUAL_INT(0x00, data[1]);  /* flags */
}

/* ---- Test 4: DEVICE_IDENTITY message with JSON payload ---- */
TEST_CASE(frame_device_identity_with_payload) {
    const char *json_payload = "{\"id\":\"nexus-001\",\"fw\":\"1.0.0\"}";
    uint16_t plen = (uint16_t)strlen(json_payload);

    uint8_t wire[FRAME_MAX_WIRE];
    size_t wire_len = build_wire_frame(0x01, 0x01, 42, 5000,
                                       (const uint8_t *)json_payload, plen,
                                       wire, sizeof(wire));
    TEST_ASSERT_TRUE(wire_len > 0);

    frame_receiver_t rx;
    frame_receiver_init(&rx);

    uint32_t result = FRAME_ERR_NONE;
    for (size_t i = 0; i < wire_len; i++) {
        result = frame_receiver_feed(&rx, wire[i]);
    }
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_NONE, result);

    uint16_t data_len = 0;
    const uint8_t *data = frame_receiver_get_data(&rx, &data_len);
    TEST_ASSERT_NOT_NULL(data);

    /* Verify header */
    TEST_ASSERT_EQUAL_INT(0x01, data[0]);
    TEST_ASSERT_EQUAL_INT(0x01, data[1]);  /* ACK_REQUIRED flag */
    TEST_ASSERT_EQUAL_INT(plen, ((uint16_t)data[8] << 8) | data[9]);

    /* Verify payload */
    TEST_ASSERT_TRUE(memcmp(data + FRAME_HEADER_SIZE, json_payload, plen) == 0);
}

/* ---- Test 5: REFLEX_DEPLOY with binary payload ---- */
TEST_CASE(frame_reflex_deploy_binary) {
    uint8_t binary_payload[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xCA, 0xFE};

    uint8_t wire[FRAME_MAX_WIRE];
    size_t wire_len = build_wire_frame(0x09, 0x00, 100, 9999,
                                       binary_payload, sizeof(binary_payload),
                                       wire, sizeof(wire));
    TEST_ASSERT_TRUE(wire_len > 0);

    frame_receiver_t rx;
    frame_receiver_init(&rx);

    uint32_t result = FRAME_ERR_NONE;
    for (size_t i = 0; i < wire_len; i++) {
        result = frame_receiver_feed(&rx, wire[i]);
    }
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_NONE, result);

    uint16_t data_len = 0;
    const uint8_t *data = frame_receiver_get_data(&rx, &data_len);
    TEST_ASSERT_NOT_NULL(data);
    TEST_ASSERT_EQUAL_INT(0x09, data[0]);

    /* Verify binary payload (after COBS, zeros are preserved) */
    TEST_ASSERT_TRUE(memcmp(data + FRAME_HEADER_SIZE, binary_payload,
                            sizeof(binary_payload)) == 0);
}

/* ---- Test 6: CRC mismatch detection ---- */
TEST_CASE(frame_crc_mismatch) {
    uint8_t wire[FRAME_MAX_WIRE];
    size_t wire_len = build_bad_crc_frame(0x05, 0x00, 1, 1000, NULL, 0,
                                          wire, sizeof(wire));
    TEST_ASSERT_TRUE(wire_len > 0);

    frame_receiver_t rx;
    frame_receiver_init(&rx);

    uint32_t result = FRAME_ERR_NONE;
    for (size_t i = 0; i < wire_len; i++) {
        result = frame_receiver_feed(&rx, wire[i]);
    }
    /* After frame complete, CRC check should fail */
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_CRC_MISMATCH, result);
    TEST_ASSERT_TRUE(rx.error_count > 0);
}

/* ---- Test 7: Oversized frame rejection ---- */
TEST_CASE(frame_oversized_rejection) {
    frame_receiver_t rx;
    frame_receiver_init(&rx);

    /* Send start delimiter to enter RECEIVING state */
    frame_receiver_feed(&rx, 0x00);
    TEST_ASSERT_EQUAL_INT(FRAME_STATE_RECEIVING, rx.state);

    /* Feed more than FRAME_MAX_COBS bytes without a delimiter */
    int found_error = 0;
    for (int i = 0; i < FRAME_MAX_COBS + 10; i++) {
        uint32_t r = frame_receiver_feed(&rx, 0x01);
        if (r == FRAME_ERR_TOO_LARGE) found_error = 1;
    }
    TEST_ASSERT_TRUE(found_error);
    TEST_ASSERT_TRUE(rx.error_count > 0);
}

/* ---- Test 8: Too-short frame rejection ---- */
TEST_CASE(frame_too_short) {
    /* Build a frame with too little data: just 2 bytes */
    uint8_t wire[FRAME_MAX_WIRE];
    uint8_t data[] = {0x05, 0x00};  /* Just msg_type and flags */
    uint8_t cobs[FRAME_MAX_COBS];
    size_t cobs_len = cobs_encode(data, 2, cobs, sizeof(cobs));

    wire[0] = 0x00;
    memcpy(wire + 1, cobs, cobs_len);
    wire[1 + cobs_len] = 0x00;
    size_t wire_len = 2 + cobs_len;

    frame_receiver_t rx;
    frame_receiver_init(&rx);

    uint32_t result = FRAME_ERR_NONE;
    for (size_t i = 0; i < wire_len; i++) {
        result = frame_receiver_feed(&rx, wire[i]);
    }
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_TOO_LARGE, result);
}

/* ---- Test 9: Multiple frames in sequence ---- */
TEST_CASE(frame_multiple_sequence) {
    frame_receiver_t rx;
    frame_receiver_init(&rx);

    /* First frame: HEARTBEAT */
    uint8_t wire1[FRAME_MAX_WIRE];
    size_t wlen1 = build_wire_frame(0x05, 0x00, 1, 1000, NULL, 0,
                                    wire1, sizeof(wire1));
    /* Second frame: SENSOR_TELEMETRY */
    uint8_t wire2[FRAME_MAX_WIRE];
    uint8_t sensor_data[] = {0x01, 0x02, 0x03};
    size_t wlen2 = build_wire_frame(0x06, 0x00, 2, 2000, sensor_data, 3,
                                    wire2, sizeof(wire2));

    /* Feed frame 1 */
    uint32_t result = FRAME_ERR_NONE;
    for (size_t i = 0; i < wlen1; i++) {
        result = frame_receiver_feed(&rx, wire1[i]);
    }
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_NONE, result);

    uint16_t dlen1 = 0;
    const uint8_t *d1 = frame_receiver_get_data(&rx, &dlen1);
    TEST_ASSERT_NOT_NULL(d1);
    TEST_ASSERT_EQUAL_INT(0x05, d1[0]);

    /* Feed frame 2 */
    result = FRAME_ERR_NONE;
    for (size_t i = 0; i < wlen2; i++) {
        result = frame_receiver_feed(&rx, wire2[i]);
    }
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_NONE, result);

    uint16_t dlen2 = 0;
    const uint8_t *d2 = frame_receiver_get_data(&rx, &dlen2);
    TEST_ASSERT_NOT_NULL(d2);
    TEST_ASSERT_EQUAL_INT(0x06, d2[0]);
}

/* ---- Test 10: frame_tx_build valid ---- */
TEST_CASE(frame_tx_build_valid) {
    uint8_t header[FRAME_HEADER_SIZE] = {0};
    header[0] = 0x05; /* HEARTBEAT */
    /* seq=0, ts=0, payload_len=0 */
    header[8] = 0x00;
    header[9] = 0x00;

    uint8_t wire[FRAME_MAX_WIRE];
    size_t wire_len = frame_tx_build(header, NULL, 0, wire, sizeof(wire));
    TEST_ASSERT_TRUE(wire_len > 0);
    TEST_ASSERT_EQUAL_INT(0x00, wire[0]);  /* Start delimiter */
    TEST_ASSERT_EQUAL_INT(0x00, wire[wire_len - 1]);  /* End delimiter */
}

/* ---- Test 11: frame_tx_build with payload ---- */
TEST_CASE(frame_tx_build_with_payload) {
    uint8_t header[FRAME_HEADER_SIZE] = {0};
    header[0] = 0x07; /* COMMAND */
    header[2] = 0x00; header[3] = 0x2A; /* seq=42 */
    header[4] = 0x00; header[5] = 0x00;
    header[6] = 0x03; header[7] = 0xE8; /* ts=1000 */
    uint8_t payload[] = {0x01, 0x02, 0x03};
    header[8] = 0x00; header[9] = 0x03; /* payload_len=3 */

    uint8_t wire[FRAME_MAX_WIRE];
    size_t wire_len = frame_tx_build(header, payload, 3, wire, sizeof(wire));
    TEST_ASSERT_TRUE(wire_len > 0);

    /* Verify it can be decoded back */
    frame_receiver_t rx;
    frame_receiver_init(&rx);
    uint32_t result = FRAME_ERR_NONE;
    for (size_t i = 0; i < wire_len; i++) {
        result = frame_receiver_feed(&rx, wire[i]);
    }
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_NONE, result);

    uint16_t dlen = 0;
    const uint8_t *data = frame_receiver_get_data(&rx, &dlen);
    TEST_ASSERT_NOT_NULL(data);
    TEST_ASSERT_EQUAL_INT(0x07, data[0]);
    TEST_ASSERT_TRUE(memcmp(data + FRAME_HEADER_SIZE, payload, 3) == 0);
}
