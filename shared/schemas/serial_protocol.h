/**
 * @file serial_protocol.h
 * @brief NEXUS Serial Protocol — C struct layouts matching serial_protocol.json schema.
 *
 * Structs for wire protocol message headers, frame info, message types,
 * and common payload formats. Field names and types match the JSON schema.
 */

#ifndef NEXUS_SERIAL_PROTOCOL_H
#define NEXUS_SERIAL_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Frame Constants
 * Matches: serial_protocol.json#/definitions/frame_info
 * =================================================================== */

#define NEXUS_FRAME_HEADER_SIZE       10
#define NEXUS_FRAME_CRC_SIZE           2
#define NEXUS_FRAME_MAX_PAYLOAD       1024
#define NEXUS_FRAME_MAX_DECODED       1036
#define NEXUS_FRAME_MAX_COBS          1051
#define NEXUS_FRAME_MAX_WIRE          1053
#define NEXUS_FRAME_DELIMITER         0x00

/* ===================================================================
 * Message Flag Bits
 * Matches: serial_protocol.json#/definitions/message_header/properties/flags
 * =================================================================== */

#define NEXUS_MSG_FLAG_ACK_REQUIRED   (1 << 0)
#define NEXUS_MSG_FLAG_IS_ACK         (1 << 1)
#define NEXUS_MSG_FLAG_IS_ERROR       (1 << 2)
#define NEXUS_MSG_FLAG_URGENT         (1 << 3)
#define NEXUS_MSG_FLAG_COMPRESSED     (1 << 4)
#define NEXUS_MSG_FLAG_ENCRYPTED      (1 << 5)
#define NEXUS_MSG_FLAG_NO_TIMESTAMP   (1 << 6)

/* ===================================================================
 * Message Directions
 * Matches: serial_protocol.json#/definitions/message_type_entry/properties/direction
 * =================================================================== */

typedef enum {
    NEXUS_DIR_N2J  = 0,  /* Node to Jetson */
    NEXUS_DIR_J2N  = 1,  /* Jetson to Node */
    NEXUS_DIR_BOTH = 2,  /* Bidirectional */
} nexus_msg_direction_t;

/* ===================================================================
 * Message Criticality
 * Matches: serial_protocol.json#/definitions/message_type_entry/properties/criticality
 * =================================================================== */

typedef enum {
    NEXUS_CRIT_TELEMETRY = 0,
    NEXUS_CRIT_COMMAND   = 1,
    NEXUS_CRIT_SAFETY    = 2,
} nexus_msg_criticality_t;

/* ===================================================================
 * Message Types (28 total)
 * Matches: firmware/wire_protocol/message.h defines
 * =================================================================== */

#define NEXUS_MSG_DEVICE_IDENTITY      0x01
#define NEXUS_MSG_ROLE_ASSIGN          0x02
#define NEXUS_MSG_ROLE_ACK             0x03
#define NEXUS_MSG_AUTO_DETECT_RESULT   0x04
#define NEXUS_MSG_HEARTBEAT            0x05
#define NEXUS_MSG_SENSOR_TELEMETRY     0x06
#define NEXUS_MSG_COMMAND              0x07
#define NEXUS_MSG_COMMAND_ACK          0x08
#define NEXUS_MSG_REFLEX_DEPLOY        0x09
#define NEXUS_MSG_REFLEX_STATUS        0x0A
#define NEXUS_MSG_FIRMWARE_QUERY       0x0B
#define NEXUS_MSG_OBS_RECORD_START     0x0C
#define NEXUS_MSG_OBS_RECORD_STOP      0x0D
#define NEXUS_MSG_OBS_DATA_CHUNK       0x0E
#define NEXUS_MSG_OBS_DATA_ACK         0x0F
#define NEXUS_MSG_SELFTEST_RESULT      0x10
#define NEXUS_MSG_ERROR_REPORT         0x11
#define NEXUS_MSG_BAUD_NEGOTIATE       0x12
#define NEXUS_MSG_BAUD_NEGOTIATE_ACK   0x13
#define NEXUS_MSG_PING                 0x14
#define NEXUS_MSG_PONG                 0x15
#define NEXUS_MSG_SAFETY_EVENT         0x1C
#define NEXUS_MSG_DEBUG_LOG            0x20
#define NEXUS_MSG_DEBUG_CMD            0x21
#define NEXUS_MSG_PARAM_GET            0x22
#define NEXUS_MSG_PARAM_SET            0x23
#define NEXUS_MSG_PARAM_RESPONSE       0x24
#define NEXUS_MSG_FIRMWARE_CHUNK       0x42
#define NEXUS_MSG_FIRMWARE_VERIFY      0x43

#define NEXUS_MSG_TYPE_COUNT 28

/* ===================================================================
 * Message Header (10 bytes, big-endian on wire)
 * Matches: serial_protocol.json#/definitions/message_header
 * =================================================================== */

typedef struct __attribute__((packed)) {
    uint8_t  msg_type;        /* Message type byte */
    uint8_t  flags;           /* Flag bits */
    uint16_t sequence;        /* Sequence number */
    uint32_t timestamp_ms;    /* Timestamp in milliseconds */
    uint16_t payload_length;  /* Payload length in bytes */
} nexus_message_header_t;

/* Verify size */
#define NEXUS_MESSAGE_HEADER_SIZE sizeof(nexus_message_header_t)

/* ===================================================================
 * Message Type Entry
 * Matches: serial_protocol.json#/definitions/message_type_entry
 * =================================================================== */

typedef struct {
    uint8_t              msg_type;      /* Message type byte */
    char                 name[24];      /* Type name (e.g. "HEARTBEAT") */
    nexus_msg_direction_t direction;    /* Allowed direction */
    nexus_msg_criticality_t criticality; /* Priority class */
} nexus_msg_type_entry_t;

/* ===================================================================
 * Frame Info
 * Matches: serial_protocol.json#/definitions/frame_info
 * =================================================================== */

typedef struct {
    int header_size;          /* 10 bytes */
    int crc_size;             /* 2 bytes */
    int max_payload;          /* 1024 bytes */
    int max_decoded_frame;    /* 1036 bytes */
    int max_cobs_frame;       /* 1051 bytes */
    int max_wire_frame;       /* 1053 bytes */
    int delimiter;            /* 0x00 */
} nexus_frame_info_t;

/* ===================================================================
 * Wire Message
 * Matches: serial_protocol.json#/definitions/wire_message
 * =================================================================== */

typedef struct {
    nexus_message_header_t header;      /* 10-byte header */
    uint8_t               payload[NEXUS_FRAME_MAX_PAYLOAD]; /* Payload data */
    uint16_t              payload_length; /* Actual payload length */
    uint16_t              crc;          /* CRC-16 checksum */
} nexus_wire_message_t;

/* ===================================================================
 * Sensor Telemetry Payload
 * Matches: serial_protocol.json#/definitions/sensor_telemetry_payload
 * =================================================================== */

typedef struct __attribute__((packed)) {
    uint8_t  sensor_id;      /* Sensor channel (0-63) */
    float    value;          /* Sensor value */
    bool     valid;          /* Whether reading is valid */
    uint32_t timestamp_ms;   /* Reading timestamp */
} nexus_sensor_telemetry_t;

/* ===================================================================
 * Command Payload
 * Matches: serial_protocol.json#/definitions/command_payload
 * =================================================================== */

typedef struct __attribute__((packed)) {
    uint8_t actuator_id;     /* Actuator channel (0-63) */
    float   value;           /* Command value */
    bool    enabled;         /* Whether actuator is enabled */
} nexus_command_payload_t;

/* ===================================================================
 * Reflex Deploy Payload
 * Matches: serial_protocol.json#/definitions/reflex_deploy_payload
 * =================================================================== */

typedef struct {
    char     name[64];       /* Reflex name */
    uint8_t  bytecode[NEXUS_FRAME_MAX_PAYLOAD]; /* Bytecode data */
    uint16_t bytecode_length; /* Bytecode length */
    float    trust_min;      /* Minimum trust required */
} nexus_reflex_deploy_t;

/* ===================================================================
 * Safety Event Payload
 * Matches: serial_protocol.json#/definitions/safety_event_payload
 * =================================================================== */

/* Safety event types */
typedef enum {
    NEXUS_SAFETY_EVT_HEARTBEAT_MISS   = 0,
    NEXUS_SAFETY_EVT_HEARTBEAT_RECOVER = 1,
    NEXUS_SAFETY_EVT_ESTOP_TRIGGERED  = 2,
    NEXUS_SAFETY_EVT_OVERCURRENT      = 3,
    NEXUS_SAFETY_EVT_WATCHDOG_TIMEOUT = 4,
    NEXUS_SAFETY_EVT_SENSOR_FAILURE   = 5,
    NEXUS_SAFETY_EVT_TRUST_VIOLATION  = 6,
    NEXUS_SAFETY_EVT_RESUME_COMMAND   = 7,
    NEXUS_SAFETY_EVT_BOOT_COMPLETE    = 8,
} nexus_safety_event_type_t;

/* Safety states */
typedef enum {
    NEXUS_STATE_NORMAL     = 0,
    NEXUS_STATE_DEGRADED   = 1,
    NEXUS_STATE_SAFE_STATE = 2,
    NEXUS_STATE_FAULT      = 3,
} nexus_safety_state_t;

typedef struct __attribute__((packed)) {
    uint8_t               event_type;  /* Safety event type */
    uint8_t               state;       /* Safety state */
    uint32_t              timestamp_ms; /* Event timestamp */
} nexus_safety_event_payload_t;

/* ===================================================================
 * Top-level Serial Protocol Config
 * Matches: serial_protocol.json (root object)
 * =================================================================== */

typedef struct {
    char                   version[8];       /* Protocol version (e.g. "1.0") */
    nexus_frame_info_t     frame;            /* Frame format constants */
    nexus_msg_type_entry_t message_types[NEXUS_MSG_TYPE_COUNT]; /* All 28 types */
    int                    num_message_types; /* Count of registered types */
    char                   crc_polynomial[24]; /* "CRC-16/CCITT-FALSE" */
} nexus_serial_protocol_t;

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_SERIAL_PROTOCOL_H */
