/**
 * @file heartbeat.h
 * @brief NEXUS Safety System - Heartbeat monitor.
 *
 * Monitors heartbeat from Jetson (100ms interval expected).
 * 3 misses -> DEGRADED, 10 misses -> SAFE_STATE.
 */

#ifndef NEXUS_HEARTBEAT_H
#define NEXUS_HEARTBEAT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HB_MISS_THRESHOLD_DEGRADED  3
#define HB_MISS_THRESHOLD_SAFE      10
#define HB_EXPECTED_INTERVAL_MS     100

typedef struct {
    uint32_t last_rx_ms;
    uint32_t missed_count;
    uint32_t total_rx;
    bool active;
} heartbeat_monitor_t;

/**
 * @brief Initialize heartbeat monitor.
 * @param hb Pointer to heartbeat monitor.
 */
void heartbeat_init(heartbeat_monitor_t *hb);

/**
 * @brief Record a heartbeat received.
 * @param hb Pointer to heartbeat monitor.
 * @param now_ms Current time in milliseconds.
 */
void heartbeat_record(heartbeat_monitor_t *hb, uint32_t now_ms);

/**
 * @brief Check heartbeat status (call periodically).
 * @param hb Pointer to heartbeat monitor.
 * @param now_ms Current time in milliseconds.
 * @return Number of missed heartbeats since last received.
 */
uint32_t heartbeat_check(heartbeat_monitor_t *hb, uint32_t now_ms);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_HEARTBEAT_H */
