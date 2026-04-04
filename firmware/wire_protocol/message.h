/**
 * @file message.h
 * @brief NEXUS Wire Protocol - Message header parsing.
 *
 * 10-byte message header (big-endian):
 *   Byte 0:    msg_type (uint8)
 *   Byte 1:    flags (uint8)
 *   Bytes 2-3: sequence_number (uint16)
 *   Bytes 4-7: timestamp_ms (uint32)
 *   Bytes 8-9: payload_length (uint16)
 */

#ifndef NEXUS_MESSAGE_H
#define NEXUS_MESSAGE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Message flag bits */
#define MSG_FLAG_ACK_REQUIRED  (1 << 0)
#define MSG_FLAG_IS_ACK        (1 << 1)
#define MSG_FLAG_IS_ERROR      (1 << 2)
#define MSG_FLAG_URGENT        (1 << 3)
#define MSG_FLAG_COMPRESSED    (1 << 4)
#define MSG_FLAG_ENCRYPTED     (1 << 5)
#define MSG_FLAG_NO_TIMESTAMP  (1 << 6)

/* Message types */
#define MSG_DEVICE_IDENTITY     0x01
#define MSG_HEARTBEAT           0x05
#define MSG_SENSOR_TELEMETRY    0x06
#define MSG_COMMAND_ACK         0x08
#define MSG_COMMAND             0x07
#define MSG_REFLEX_DEPLOY       0x09
#define MSG_SAFETY_EVENT        0x1C

typedef struct __attribute__((packed)) {
    uint8_t  msg_type;
    uint8_t  flags;
    uint16_t sequence;
    uint32_t timestamp_ms;
    uint16_t payload_length;
} nexus_message_header_t;

/**
 * @brief Parse a message header from raw bytes.
 * @param data Pointer to 10-byte header data.
 * @param header Output: parsed header.
 * @return true if parsed successfully, false on error.
 */
bool message_parse_header(const uint8_t *data, nexus_message_header_t *header);

/**
 * @brief Get the message type name string.
 * @param msg_type Message type byte.
 * @return Static string with the type name.
 */
const char *message_type_name(uint8_t msg_type);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_MESSAGE_H */
