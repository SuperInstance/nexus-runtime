/**
 * @file frame.c
 * @brief NEXUS Wire Protocol - Frame reception state machine (stub).
 */

#include "frame.h"
#include <string.h>

void frame_receiver_init(frame_receiver_t *rx) {
    if (!rx) {
        return;
    }
    memset(rx, 0, sizeof(frame_receiver_t));
    rx->state = FRAME_STATE_IDLE;
    rx->cobs_idx = 0;
    rx->decode_len = 0;
    rx->error_count = 0;
}

uint32_t frame_receiver_feed(frame_receiver_t *rx, uint8_t byte) {
    if (!rx) {
        return FRAME_ERR_BUFFER_OVERFLOW;
    }

    /* TODO: Implement full frame reception state machine */
    /* IDLE: wait for 0x00 delimiter, transition to RECEIVING */
    /* RECEIVING: accumulate bytes until 0x00 delimiter */
    /* On frame complete: COBS decode, CRC validate */

    (void)byte;
    return FRAME_ERR_NONE;
}

const uint8_t *frame_receiver_get_data(const frame_receiver_t *rx, uint16_t *out_len) {
    if (!rx || !out_len) {
        return NULL;
    }
    *out_len = rx->decode_len;
    return rx->decoded;
}
