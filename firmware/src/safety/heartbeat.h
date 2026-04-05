/**
 * @file heartbeat.h
 * @brief NEXUS Safety System — Heartbeat monitor.
 *
 * Monitors heartbeat from Jetson (configurable interval).
 * Tracks miss count and provides status (OK, DEGRADED, SAFE, OFFLINE).
 */

#ifndef NEXUS_HEARTBEAT_H
#define NEXUS_HEARTBEAT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HB_OK,
    HB_DEGRADED,
    HB_SAFE,
    HB_OFFLINE,
} heartbeat_status_t;

typedef struct {
    uint32_t last_heartbeat_ms;
    uint32_t expected_interval_ms;    /* 1000ms from node, 5000ms from Jetson */
    uint32_t degrade_threshold_ms;    /* 5000ms */
    uint32_t safe_threshold_ms;       /* 10000ms */
    uint32_t miss_count;
    uint32_t total_misses;
    bool online;
} heartbeat_monitor_t;

/**
 * @brief Initialize heartbeat monitor with configurable interval.
 */
void heartbeat_init(heartbeat_monitor_t *hb, uint32_t expected_interval_ms);

/**
 * @brief Record a heartbeat received.
 */
void heartbeat_record(heartbeat_monitor_t *hb, uint32_t tick_ms);

/**
 * @brief Check heartbeat status based on elapsed time.
 * @return Current heartbeat status.
 */
heartbeat_status_t heartbeat_check(heartbeat_monitor_t *hb, uint32_t tick_ms);

/**
 * @brief Get current miss count.
 */
uint32_t heartbeat_get_miss_count(heartbeat_monitor_t *hb);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_HEARTBEAT_H */
