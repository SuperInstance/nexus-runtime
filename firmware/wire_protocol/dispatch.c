/**
 * @file dispatch.c
 * @brief NEXUS Wire Protocol - Message type dispatch implementation.
 */

#include "dispatch.h"
#include <string.h>

void dispatch_init(dispatch_table_t *table) {
    if (!table) return;
    memset(table, 0, sizeof(dispatch_table_t));
    table->count = 0;
}

bool dispatch_register(dispatch_table_t *table, uint8_t msg_type,
                       dispatch_handler_t handler, void *user_ctx) {
    if (!table || !handler) return false;

    /* Check for existing entry to update */
    for (uint8_t i = 0; i < table->count; i++) {
        if (table->entries[i].msg_type == msg_type && table->entries[i].active) {
            table->entries[i].handler = handler;
            table->entries[i].user_ctx = user_ctx;
            return true;
        }
    }

    /* Add new entry */
    if (table->count >= DISPATCH_MAX_HANDLERS) return false;

    table->entries[table->count].msg_type = msg_type;
    table->entries[table->count].handler = handler;
    table->entries[table->count].user_ctx = user_ctx;
    table->entries[table->count].active = true;
    table->count++;
    return true;
}

void dispatch_unregister(dispatch_table_t *table, uint8_t msg_type) {
    if (!table) return;
    for (uint8_t i = 0; i < table->count; i++) {
        if (table->entries[i].msg_type == msg_type && table->entries[i].active) {
            table->entries[i].active = false;
            return;
        }
    }
}

bool dispatch_message(dispatch_table_t *table,
                      const nexus_message_header_t *header,
                      const uint8_t *payload, uint16_t payload_len) {
    if (!table || !header) return false;

    for (uint8_t i = 0; i < table->count; i++) {
        if (table->entries[i].active && table->entries[i].msg_type == header->msg_type) {
            return table->entries[i].handler(header, payload, payload_len,
                                              table->entries[i].user_ctx);
        }
    }

    return false;
}
