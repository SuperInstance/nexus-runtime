/**
 * @file message.c
 * @brief NEXUS Wire Protocol - Message header parsing (stub).
 */

#include "message.h"
#include <string.h>

bool message_parse_header(const uint8_t *data, nexus_message_header_t *header) {
    if (!data || !header) {
        return false;
    }

    /* Parse big-endian fields */
    header->msg_type       = data[0];
    header->flags          = data[1];
    header->sequence       = ((uint16_t)data[2] << 8) | data[3];
    header->timestamp_ms   = ((uint32_t)data[4] << 24) | ((uint32_t)data[5] << 16) |
                             ((uint32_t)data[6] << 8)  | data[7];
    header->payload_length = ((uint16_t)data[8] << 8) | data[9];

    return true;
}

const char *message_type_name(uint8_t msg_type) {
    switch (msg_type) {
        case MSG_DEVICE_IDENTITY:  return "DEVICE_IDENTITY";
        case MSG_HEARTBEAT:        return "HEARTBEAT";
        case MSG_SENSOR_TELEMETRY: return "SENSOR_TELEMETRY";
        case MSG_COMMAND_ACK:      return "COMMAND_ACK";
        case MSG_COMMAND:          return "COMMAND";
        case MSG_REFLEX_DEPLOY:    return "REFLEX_DEPLOY";
        case MSG_SAFETY_EVENT:     return "SAFETY_EVENT";
        default:                   return "UNKNOWN";
    }
}
