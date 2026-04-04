/**
 * @file dispatch.h
 * @brief NEXUS Wire Protocol - Message type dispatch.
 */

#ifndef NEXUS_DISPATCH_H
#define NEXUS_DISPATCH_H

#include <stdint.h>
#include <stdbool.h>
#include "message.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dispatch a received message to the appropriate handler.
 * @param header Parsed message header.
 * @param payload Pointer to message payload data.
 * @param payload_len Length of payload in bytes.
 * @return true if message was handled, false if unhandled.
 */
bool dispatch_message(const nexus_message_header_t *header,
                      const uint8_t *payload, uint16_t payload_len);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_DISPATCH_H */
