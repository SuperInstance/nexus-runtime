/**
 * @file watchdog.c
 * @brief NEXUS Safety System - Hardware/software watchdog (stub).
 */

#include "watchdog.h"
#include <string.h>

void watchdog_init(watchdog_state_t *wdt) {
    if (!wdt) {
        return;
    }
    memset(wdt, 0, sizeof(watchdog_state_t));
    wdt->timeout_ms = 1000;
    wdt->initialized = true;
}

void watchdog_feed_hw(void) {
    /* TODO: Implement 0x55/0xAA alternating pattern on WDT_KICK_PIN */
}

bool watchdog_register_task(watchdog_state_t *wdt, uint8_t task_index, uint32_t timeout_ms) {
    if (!wdt || task_index >= WDT_MAX_MONITORED_TASKS) {
        return false;
    }
    wdt->last_checkin[task_index] = 0;
    (void)timeout_ms;
    return true;
}

void watchdog_task_checkin(watchdog_state_t *wdt, uint8_t task_index, uint32_t now_ms) {
    if (!wdt || task_index >= WDT_MAX_MONITORED_TASKS) {
        return;
    }
    wdt->last_checkin[task_index] = now_ms;
}
