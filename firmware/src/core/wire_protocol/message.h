/**
 * @file message.h
 * @brief NEXUS Wire Protocol - Message types and header parsing.
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

/* Message criticality levels */
#define MSG_CRIT_TELEMETRY 0
#define MSG_CRIT_COMMAND   1
#define MSG_CRIT_SAFETY    2

/* Message directions */
#define MSG_DIR_N2J 0  /* Node to Jetson */
#define MSG_DIR_J2N 1  /* Jetson to Node */
#define MSG_DIR_BOTH 2 /* Bidirectional */

/* All 28 message types */
#define MSG_DEVICE_IDENTITY      0x01
#define MSG_ROLE_ASSIGN          0x02
#define MSG_ROLE_ACK             0x03
#define MSG_AUTO_DETECT_RESULT   0x04
#define MSG_HEARTBEAT            0x05
#define MSG_SENSOR_TELEMETRY     0x06
#define MSG_COMMAND              0x07
#define MSG_COMMAND_ACK          0x08
#define MSG_REFLEX_DEPLOY        0x09
#define MSG_REFLEX_STATUS        0x0A
#define MSG_FIRMWARE_QUERY       0x0B
#define MSG_OBS_RECORD_START     0x0C
#define MSG_OBS_RECORD_STOP      0x0D
#define MSG_OBS_DATA_CHUNK       0x0E
#define MSG_OBS_DATA_ACK         0x0F
#define MSG_SELFTEST_RESULT      0x10
#define MSG_ERROR_REPORT         0x11
#define MSG_BAUD_NEGOTIATE       0x12
#define MSG_BAUD_NEGOTIATE_ACK   0x13
#define MSG_PING                 0x14
#define MSG_PONG                 0x15
#define MSG_SAFETY_EVENT         0x1C
#define MSG_DEBUG_LOG            0x20
#define MSG_DEBUG_CMD            0x21
#define MSG_PARAM_GET            0x22
#define MSG_PARAM_SET            0x23
#define MSG_PARAM_RESPONSE       0x24
#define MSG_FIRMWARE_CHUNK       0x42
#define MSG_FIRMWARE_VERIFY      0x43

typedef struct __attribute__((packed)) {
    uint8_t  msg_type;
    uint8_t  flags;
    uint16_t sequence;
    uint32_t timestamp_ms;
    uint16_t payload_length;
} nexus_message_header_t;

/* Message type info for lookup */
typedef struct {
    uint8_t  msg_type;
    const char *name;
    uint8_t  direction;    /* MSG_DIR_N2J, MSG_DIR_J2N, MSG_DIR_BOTH */
    uint8_t  criticality;  /* MSG_CRIT_TELEMETRY, MSG_CRIT_COMMAND, MSG_CRIT_SAFETY */
} msg_type_info_t;

/**
 * @brief Get the total number of defined message types.
 * @return Number of message types (28).
 */
uint8_t message_type_count(void);

/**
 * @brief Get info about a message type by index.
 * @param index Index (0 to message_type_count()-1).
 * @return Pointer to type info, or NULL if index out of range.
 */
const msg_type_info_t *message_type_info(uint8_t index);

/**
 * @brief Find message type info by type ID.
 * @param msg_type Message type byte.
 * @return Pointer to type info, or NULL if unknown.
 */
const msg_type_info_t *message_type_lookup(uint8_t msg_type);

/**
 * @brief Parse a message header from raw big-endian bytes.
 * @param data Pointer to 10-byte header data.
 * @param header Output: parsed header.
 * @return true if parsed successfully, false on error.
 */
bool message_parse_header(const uint8_t *data, nexus_message_header_t *header);

/**
 * @brief Serialize a message header to big-endian bytes.
 * @param header Pointer to header struct.
 * @param out Output buffer (must be >= 10 bytes).
 * @return true if serialized successfully.
 */
bool message_serialize_header(const nexus_message_header_t *header, uint8_t *out);

/**
 * @brief Get the message type name string.
 * @param msg_type Message type byte.
 * @return Static string with the type name, or "UNKNOWN" if not found.
 */
const char *message_type_name(uint8_t msg_type);

/**
 * @brief Check if a message type is valid.
 * @param msg_type Message type byte.
 * @return true if the type is defined.
 */
bool message_type_is_valid(uint8_t msg_type);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_MESSAGE_H */
