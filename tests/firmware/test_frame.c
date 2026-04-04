/**
 * @file test_frame.c
 * @brief NEXUS frame reception tests (stub).
 */

#include "unity.h"
#include "frame.h"

TEST_CASE(frame_init) {
    frame_receiver_t rx;
    frame_receiver_init(&rx);
    TEST_ASSERT_EQUAL_INT(FRAME_STATE_IDLE, rx.state);
    TEST_ASSERT_EQUAL_INT(0, rx.cobs_idx);
    TEST_ASSERT_EQUAL_INT(0, rx.error_count);
}

TEST_CASE(frame_feed_null) {
    uint32_t err = frame_receiver_feed(NULL, 0x00);
    TEST_ASSERT_EQUAL_INT(FRAME_ERR_BUFFER_OVERFLOW, err);
}
