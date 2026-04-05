/**
 * @file dispatch.h
 * @brief NEXUS Wire Protocol - Message type dispatch with handler registration.
 */

#ifndef NEXUS_DISPATCH_H
#define NEXUS_DISPATCH_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "message.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of registered handlers */
#define DISPATCH_MAX_HANDLERS 32

/**
 * @brief Handler callback function signature.
 *
 * @param header Parsed message header.
 * @param payload Pointer to message payload data.
 * @param payload_len Length of payload in bytes.
 * @param user_ctx User-provided context pointer.
 * @return true if message was handled successfully.
 */
typedef bool (*dispatch_handler_t)(const nexus_message_header_t *header,
                                   const uint8_t *payload,
                                   uint16_t payload_len,
                                   void *user_ctx);

typedef struct {
    uint8_t msg_type;
    dispatch_handler_t handler;
    void *user_ctx;
    bool active;
} dispatch_entry_t;

typedef struct {
    dispatch_entry_t entries[DISPATCH_MAX_HANDLERS];
    uint8_t count;
} dispatch_table_t;

/**
 * @brief Initialize dispatch table.
 * @param table Pointer to dispatch table.
 */
void dispatch_init(dispatch_table_t *table);

/**
 * @brief Register a handler for a message type.
 * @param table Pointer to dispatch table.
 * @param msg_type Message type to handle.
 * @param handler Handler callback function.
 * @param user_ctx User context passed to handler (can be NULL).
 * @return true if registered, false if table is full.
 */
bool dispatch_register(dispatch_table_t *table, uint8_t msg_type,
                       dispatch_handler_t handler, void *user_ctx);

/**
 * @brief Unregister handler for a message type.
 * @param table Pointer to dispatch table.
 * @param msg_type Message type to unregister.
 */
void dispatch_unregister(dispatch_table_t *table, uint8_t msg_type);

/**
 * @brief Dispatch a received message to the appropriate handler.
 * @param table Pointer to dispatch table.
 * @param header Parsed message header.
 * @param payload Pointer to message payload data.
 * @param payload_len Length of payload in bytes.
 * @return true if message was handled, false if unhandled.
 */
bool dispatch_message(dispatch_table_t *table,
                      const nexus_message_header_t *header,
                      const uint8_t *payload, uint16_t payload_len);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_DISPATCH_H */
