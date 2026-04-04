/**
 * @file watchdog.h
 * @brief NEXUS Safety System - Hardware/software watchdog.
 *
 * HW watchdog: MAX6818, alternating 0x55/0xAA pattern, 1.0s timeout.
 * SW watchdog: FreeRTOS task monitoring, 1.0s timeout per task.
 */

#ifndef NEXUS_WATCHDOG_H
#define NEXUS_WATCHDOG_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WDT_MAX_MONITORED_TASKS 8

typedef struct {
    uint32_t last_checkin[WDT_MAX_MONITORED_TASKS];
    uint32_t timeout_ms;
    bool initialized;
} watchdog_state_t;

/**
 * @brief Initialize the watchdog system.
 * @param wdt Pointer to watchdog state.
 */
void watchdog_init(watchdog_state_t *wdt);

/**
 * @brief Feed the hardware watchdog (0x55/0xAA pattern).
 */
void watchdog_feed_hw(void);

/**
 * @brief Register a task for software watchdog monitoring.
 * @param wdt Pointer to watchdog state.
 * @param task_index Task index (0 to WDT_MAX_MONITORED_TASKS-1).
 * @param timeout_ms Timeout in milliseconds.
 * @return true if registered, false on error.
 */
bool watchdog_register_task(watchdog_state_t *wdt, uint8_t task_index, uint32_t timeout_ms);

/**
 * @brief Task check-in (call periodically from monitored task).
 * @param wdt Pointer to watchdog state.
 * @param task_index Task index.
 * @param now_ms Current time in milliseconds.
 */
void watchdog_task_checkin(watchdog_state_t *wdt, uint8_t task_index, uint32_t now_ms);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_WATCHDOG_H */
