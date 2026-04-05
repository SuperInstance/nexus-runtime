/**
 * @file heartbeat.c
 * @brief NEXUS Safety System — Heartbeat monitor implementation.
 *
 * Monitors heartbeat from Jetson/node. Computes missed heartbeats
 * based on elapsed time and expected interval.
 */

#include "heartbeat.h"
#include <string.h>

void heartbeat_init(heartbeat_monitor_t *hb, uint32_t expected_interval_ms) {
    if (!hb) return;
    memset(hb, 0, sizeof(heartbeat_monitor_t));
    hb->expected_interval_ms = expected_interval_ms;
    hb->degrade_threshold_ms = 5000;   /* 5 seconds */
    hb->safe_threshold_ms = 10000;     /* 10 seconds */
    hb->online = false;
}

void heartbeat_record(heartbeat_monitor_t *hb, uint32_t tick_ms) {
    if (!hb) return;
    hb->last_heartbeat_ms = tick_ms;
    hb->miss_count = 0;
    hb->online = true;
}

heartbeat_status_t heartbeat_check(heartbeat_monitor_t *hb, uint32_t tick_ms) {
    if (!hb) return HB_OFFLINE;
    if (!hb->online) return HB_OFFLINE;

    uint32_t elapsed = tick_ms - hb->last_heartbeat_ms;

    /* Update miss count based on elapsed time */
    if (hb->expected_interval_ms > 0) {
        hb->miss_count = elapsed / hb->expected_interval_ms;
    }

    /* Count total misses for logging */
    if (hb->miss_count > 0) {
        hb->total_misses += hb->miss_count;
    }

    if (elapsed >= hb->safe_threshold_ms) {
        return HB_SAFE;
    }
    if (elapsed >= hb->degrade_threshold_ms) {
        return HB_DEGRADED;
    }
    return HB_OK;
}

uint32_t heartbeat_get_miss_count(heartbeat_monitor_t *hb) {
    if (!hb) return 0;
    return hb->miss_count;
}
