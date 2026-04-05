/**
 * @file frame.c
 * @brief NEXUS Wire Protocol - Frame reception state machine and TX builder.
 *
 * Wire format: [0x00] [COBS(header + payload + CRC)] [0x00]
 * The COBS-encoded region contains:
 *   [10-byte Message Header] [Payload: 0-1024 bytes] [2-byte CRC-16]
 */

#include "frame.h"
#include "cobs.h"
#include "crc16.h"
#include <string.h>

void frame_receiver_init(frame_receiver_t *rx) {
    if (!rx) return;
    memset(rx, 0, sizeof(frame_receiver_t));
    rx->state = FRAME_STATE_IDLE;
    rx->cobs_idx = 0;
    rx->decode_len = 0;
    rx->error_count = 0;
}

uint32_t frame_receiver_feed(frame_receiver_t *rx, uint8_t byte) {
    if (!rx) return FRAME_ERR_BUFFER_OVERFLOW;

    switch (rx->state) {
    case FRAME_STATE_IDLE:
        if (byte == 0x00) {
            /* Start of frame delimiter detected */
            rx->state = FRAME_STATE_RECEIVING;
            rx->cobs_idx = 0;
        }
        /* Ignore non-delimiter bytes in IDLE */
        return FRAME_ERR_NONE;

    case FRAME_STATE_RECEIVING:
        if (byte == 0x00) {
            /* End of frame delimiter detected */
            if (rx->cobs_idx == 0) {
                /* Empty frame (consecutive delimiters), back to IDLE */
                rx->state = FRAME_STATE_IDLE;
                return FRAME_ERR_NONE;
            }

            rx->state = FRAME_STATE_IDLE;

            /* COBS decode */
            size_t dec_len = cobs_decode(rx->cobs_buffer, rx->cobs_idx,
                                         rx->decoded, FRAME_MAX_DECODED);
            if (dec_len == 0 || dec_len > FRAME_MAX_DECODED) {
                rx->error_count++;
                return FRAME_ERR_TOO_LARGE;
            }

            /* Minimum frame: 10-byte header + 2-byte CRC = 12 bytes */
            if (dec_len < FRAME_HEADER_SIZE + FRAME_CRC_SIZE) {
                rx->error_count++;
                return FRAME_ERR_TOO_LARGE;
            }

            /* Extract and verify CRC */
            uint16_t payload_len = ((uint16_t)rx->decoded[8] << 8) | rx->decoded[9];

            /* Validate payload length field */
            size_t expected_len = (size_t)FRAME_HEADER_SIZE + payload_len + FRAME_CRC_SIZE;
            if (expected_len != dec_len) {
                rx->error_count++;
                return FRAME_ERR_TOO_LARGE;
            }

            /* CRC covers header + payload (everything except the 2 CRC bytes at the end) */
            uint16_t received_crc = ((uint16_t)rx->decoded[dec_len - 2] << 8) |
                                    rx->decoded[dec_len - 1];
            uint16_t computed_crc = crc16_ccitt(rx->decoded, dec_len - FRAME_CRC_SIZE);

            if (received_crc != computed_crc) {
                rx->error_count++;
                return FRAME_ERR_CRC_MISMATCH;
            }

            /* Frame is valid */
            rx->decode_len = (uint16_t)dec_len;
            return FRAME_ERR_NONE;
        } else {
            /* Accumulate COBS-encoded byte */
            if (rx->cobs_idx >= FRAME_MAX_COBS) {
                /* Frame too large, discard and return to IDLE */
                rx->state = FRAME_STATE_IDLE;
                rx->cobs_idx = 0;
                rx->error_count++;
                return FRAME_ERR_TOO_LARGE;
            }
            rx->cobs_buffer[rx->cobs_idx++] = byte;
            return FRAME_ERR_NONE;
        }

    default:
        rx->state = FRAME_STATE_IDLE;
        return FRAME_ERR_BUFFER_OVERFLOW;
    }
}

const uint8_t *frame_receiver_get_data(const frame_receiver_t *rx, uint16_t *out_len) {
    if (!rx || !out_len) return NULL;
    *out_len = rx->decode_len;
    return rx->decoded;
}

size_t frame_tx_build(const uint8_t *header, const uint8_t *payload,
                      uint16_t payload_len, uint8_t *out, size_t out_max) {
    if (!header || !out) return 0;
    if (out_max < FRAME_MAX_WIRE) return 0;
    if (payload_len > FRAME_MAX_PAYLOAD) return 0;

    /* Build the data to COBS-encode: header + payload + CRC */
    uint8_t encode_buf[FRAME_MAX_DECODED];
    size_t data_len = FRAME_HEADER_SIZE + payload_len;

    memcpy(encode_buf, header, FRAME_HEADER_SIZE);
    if (payload_len > 0 && payload) {
        memcpy(encode_buf + FRAME_HEADER_SIZE, payload, payload_len);
    }

    /* Compute CRC over header + payload */
    uint16_t crc = crc16_ccitt(encode_buf, data_len);
    encode_buf[data_len] = (uint8_t)(crc >> 8);
    encode_buf[data_len + 1] = (uint8_t)(crc & 0xFF);
    data_len += FRAME_CRC_SIZE;

    /* COBS encode */
    uint8_t cobs_buf[FRAME_MAX_COBS];
    size_t cobs_len = cobs_encode(encode_buf, data_len, cobs_buf, sizeof(cobs_buf));
    if (cobs_len == 0) return 0;

    /* Build wire frame: [0x00] [COBS] [0x00] */
    size_t wire_len = 1 + cobs_len + 1;
    if (wire_len > out_max) return 0;

    out[0] = 0x00;
    memcpy(out + 1, cobs_buf, cobs_len);
    out[1 + cobs_len] = 0x00;

    return wire_len;
}
