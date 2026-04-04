/**
 * @file frame.h
 * @brief NEXUS Wire Protocol - Frame reception state machine.
 *
 * States: IDLE, RECEIVING
 * Maximum decoded frame: 1036 bytes (10 header + 1024 payload + 2 CRC)
 * Maximum COBS-encoded frame: 1051 bytes
 * Maximum wire frame: 1053 bytes (1 + 1051 + 1 delimiters)
 */

#ifndef NEXUS_FRAME_H
#define NEXUS_FRAME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRAME_MAX_DECODED   1036
#define FRAME_MAX_COBS      1051
#define FRAME_MAX_WIRE      1053
#define FRAME_HEADER_SIZE   10
#define FRAME_CRC_SIZE      2
#define FRAME_MAX_PAYLOAD   1024

#define FRAME_ERR_NONE              0
#define FRAME_ERR_TOO_LARGE         0x5001
#define FRAME_ERR_CRC_MISMATCH      0x5003
#define FRAME_ERR_BUFFER_OVERFLOW   0x5004

typedef enum {
    FRAME_STATE_IDLE = 0,
    FRAME_STATE_RECEIVING = 1,
} frame_state_t;

typedef struct {
    frame_state_t state;
    uint8_t cobs_buffer[FRAME_MAX_COBS];
    uint8_t decoded[FRAME_MAX_DECODED];
    uint16_t cobs_idx;
    uint16_t decode_len;
    uint32_t error_count;
} frame_receiver_t;

/**
 * @brief Initialize frame receiver.
 * @param rx Pointer to receiver state.
 */
void frame_receiver_init(frame_receiver_t *rx);

/**
 * @brief Feed one byte into the frame receiver state machine.
 * @param rx Pointer to receiver state.
 * @param byte Received byte.
 * @return Frame error code (FRAME_ERR_NONE when a complete frame is ready).
 */
uint32_t frame_receiver_feed(frame_receiver_t *rx, uint8_t byte);

/**
 * @brief Get the decoded frame data.
 * @param rx Pointer to receiver state.
 * @param out_len Output: decoded frame length.
 * @return Pointer to decoded frame data, or NULL if no frame ready.
 */
const uint8_t *frame_receiver_get_data(const frame_receiver_t *rx, uint16_t *out_len);

/**
 * @brief Build a wire frame from header + payload.
 *
 * Wire format: [0x00] [COBS(header + payload + CRC)] [0x00]
 *
 * @param header Pointer to 10-byte message header.
 * @param payload Pointer to payload data.
 * @param payload_len Length of payload.
 * @param out Output buffer (must be >= FRAME_MAX_WIRE bytes).
 * @param out_max Maximum output buffer size.
 * @return Total wire frame length, or 0 on error.
 */
size_t frame_tx_build(const uint8_t *header, const uint8_t *payload,
                      uint16_t payload_len, uint8_t *out, size_t out_max);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_FRAME_H */
