/**
 * @file message.c
 * @brief NEXUS Wire Protocol - Message type definitions and header parsing.
 */

#include "message.h"
#include <string.h>

/* Static table of all 28 message types */
static const msg_type_info_t msg_types[] = {
    { 0x01, "DEVICE_IDENTITY",    MSG_DIR_N2J,  MSG_CRIT_TELEMETRY },
    { 0x02, "ROLE_ASSIGN",        MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x03, "ROLE_ACK",           MSG_DIR_N2J,  MSG_CRIT_COMMAND   },
    { 0x04, "AUTO_DETECT_RESULT", MSG_DIR_N2J,  MSG_CRIT_TELEMETRY },
    { 0x05, "HEARTBEAT",          MSG_DIR_BOTH, MSG_CRIT_TELEMETRY },
    { 0x06, "SENSOR_TELEMETRY",   MSG_DIR_N2J,  MSG_CRIT_TELEMETRY },
    { 0x07, "COMMAND",            MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x08, "COMMAND_ACK",        MSG_DIR_N2J,  MSG_CRIT_COMMAND   },
    { 0x09, "REFLEX_DEPLOY",      MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x0A, "REFLEX_STATUS",      MSG_DIR_N2J,  MSG_CRIT_TELEMETRY },
    { 0x0B, "FIRMWARE_QUERY",     MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x0C, "OBS_RECORD_START",   MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x0D, "OBS_RECORD_STOP",    MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x0E, "OBS_DATA_CHUNK",     MSG_DIR_N2J,  MSG_CRIT_TELEMETRY },
    { 0x0F, "OBS_DATA_ACK",       MSG_DIR_J2N,  MSG_CRIT_TELEMETRY },
    { 0x10, "SELFTEST_RESULT",    MSG_DIR_N2J,  MSG_CRIT_TELEMETRY },
    { 0x11, "ERROR_REPORT",       MSG_DIR_N2J,  MSG_CRIT_SAFETY    },
    { 0x12, "BAUD_NEGOTIATE",     MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x13, "BAUD_NEGOTIATE_ACK", MSG_DIR_N2J,  MSG_CRIT_COMMAND   },
    { 0x14, "PING",               MSG_DIR_BOTH, MSG_CRIT_TELEMETRY },
    { 0x15, "PONG",               MSG_DIR_BOTH, MSG_CRIT_TELEMETRY },
    { 0x1C, "SAFETY_EVENT",       MSG_DIR_N2J,  MSG_CRIT_SAFETY    },
    { 0x20, "DEBUG_LOG",          MSG_DIR_N2J,  MSG_CRIT_TELEMETRY },
    { 0x21, "DEBUG_CMD",          MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x22, "PARAM_GET",          MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x23, "PARAM_SET",          MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x24, "PARAM_RESPONSE",     MSG_DIR_N2J,  MSG_CRIT_TELEMETRY },
    { 0x42, "FIRMWARE_CHUNK",     MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
    { 0x43, "FIRMWARE_VERIFY",    MSG_DIR_J2N,  MSG_CRIT_COMMAND   },
};

#define MSG_TYPE_COUNT (sizeof(msg_types) / sizeof(msg_types[0]))

uint8_t message_type_count(void) {
    return (uint8_t)MSG_TYPE_COUNT;
}

const msg_type_info_t *message_type_info(uint8_t index) {
    if (index >= MSG_TYPE_COUNT) return NULL;
    return &msg_types[index];
}

const msg_type_info_t *message_type_lookup(uint8_t msg_type) {
    for (size_t i = 0; i < MSG_TYPE_COUNT; i++) {
        if (msg_types[i].msg_type == msg_type) {
            return &msg_types[i];
        }
    }
    return NULL;
}

bool message_parse_header(const uint8_t *data, nexus_message_header_t *header) {
    if (!data || !header) return false;

    header->msg_type       = data[0];
    header->flags          = data[1];
    header->sequence       = ((uint16_t)data[2] << 8) | data[3];
    header->timestamp_ms   = ((uint32_t)data[4] << 24) | ((uint32_t)data[5] << 16) |
                             ((uint32_t)data[6] << 8)  | data[7];
    header->payload_length = ((uint16_t)data[8] << 8) | data[9];

    return true;
}

bool message_serialize_header(const nexus_message_header_t *header, uint8_t *out) {
    if (!header || !out) return false;

    out[0] = header->msg_type;
    out[1] = header->flags;
    out[2] = (uint8_t)(header->sequence >> 8);
    out[3] = (uint8_t)(header->sequence & 0xFF);
    out[4] = (uint8_t)(header->timestamp_ms >> 24);
    out[5] = (uint8_t)(header->timestamp_ms >> 16);
    out[6] = (uint8_t)(header->timestamp_ms >> 8);
    out[7] = (uint8_t)(header->timestamp_ms & 0xFF);
    out[8] = (uint8_t)(header->payload_length >> 8);
    out[9] = (uint8_t)(header->payload_length & 0xFF);

    return true;
}

const char *message_type_name(uint8_t msg_type) {
    const msg_type_info_t *info = message_type_lookup(msg_type);
    if (info) return info->name;
    return "UNKNOWN";
}

bool message_type_is_valid(uint8_t msg_type) {
    return message_type_lookup(msg_type) != NULL;
}
