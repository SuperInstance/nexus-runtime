/**
 * @file watchdog.h
 * @brief NEXUS Safety System — Hardware watchdog timer.
 *
 * HW watchdog: MAX6818-style, alternating 0x55/0xAA kick pattern,
 * 1000ms timeout. 200ms recommended kick interval.
 */

#ifndef NEXUS_WATCHDOG_H
#define NEXUS_WATCHDOG_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t kick_pattern;       /* Alternating 0x55/0xAA */
    uint32_t last_kick_ms;
    uint32_t timeout_ms;        /* 1000ms hardware timeout */
    uint32_t kick_interval_ms;  /* 200ms kick interval */
    bool enabled;
    bool triggered;
    uint32_t trigger_count;
} watchdog_t;

/**
 * @brief Initialize watchdog with defaults (1000ms timeout, 200ms interval).
 */
void watchdog_init(watchdog_t *wdt);

/**
 * @brief Kick the watchdog with alternating 0x55/0xAA pattern.
 */
void watchdog_kick(watchdog_t *wdt, uint32_t tick_ms);

/**
 * @brief Check if watchdog has expired.
 * @return true if watchdog has timed out since last kick.
 */
bool watchdog_check(watchdog_t *wdt, uint32_t tick_ms);

/**
 * @brief Disable watchdog (TESTING ONLY — NEVER in production!).
 */
void watchdog_disable(watchdog_t *wdt);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_WATCHDOG_H */
