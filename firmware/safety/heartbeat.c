/**
 * @file heartbeat.c
 * @brief NEXUS Safety System - Heartbeat monitor (stub).
 */

#include "heartbeat.h"
#include <string.h>

void heartbeat_init(heartbeat_monitor_t *hb) {
    if (!hb) {
        return;
    }
    memset(hb, 0, sizeof(heartbeat_monitor_t));
    hb->active = false;
}

void heartbeat_record(heartbeat_monitor_t *hb, uint32_t now_ms) {
    if (!hb) {
        return;
    }
    hb->last_rx_ms = now_ms;
    hb->missed_count = 0;
    hb->total_rx++;
    hb->active = true;
}

uint32_t heartbeat_check(heartbeat_monitor_t *hb, uint32_t now_ms) {
    if (!hb || !hb->active) {
        return 0;
    }

    /* TODO: Compute missed count based on elapsed time */
    (void)now_ms;
    return hb->missed_count;
}
