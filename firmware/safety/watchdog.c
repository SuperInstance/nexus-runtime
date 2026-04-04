/**
 * @file watchdog.c
 * @brief NEXUS Safety System — Watchdog implementation.
 *
 * Alternating 0x55/0xAA kick pattern for MAX6818-style hardware watchdog.
 * On host: simulates timeout tracking for testing.
 */

#include "watchdog.h"
#include <string.h>

void watchdog_init(watchdog_t *wdt) {
    if (!wdt) return;
    memset(wdt, 0, sizeof(watchdog_t));
    wdt->kick_pattern = 0x55;    /* First kick must be 0x55 */
    wdt->timeout_ms = 1000;      /* 1.0 second hardware timeout */
    wdt->kick_interval_ms = 200; /* Kick every 200ms */
    wdt->enabled = true;
    wdt->triggered = false;
    wdt->trigger_count = 0;
}

void watchdog_kick(watchdog_t *wdt, uint32_t tick_ms) {
    if (!wdt || !wdt->enabled) return;

    /* Validate alternating pattern: expect opposite of current */
    uint8_t expected = (wdt->kick_pattern == 0x55) ? 0xAA : 0x55;

    /* On real hardware, we would write the pattern to the WDT pin.
     * In simulation, we just track the pattern alternation. */
    wdt->kick_pattern = expected;
    wdt->last_kick_ms = tick_ms;

    /* Clear triggered flag on successful kick */
    wdt->triggered = false;
}

bool watchdog_check(watchdog_t *wdt, uint32_t tick_ms) {
    if (!wdt || !wdt->enabled) return false;

    uint32_t elapsed = tick_ms - wdt->last_kick_ms;
    /* Handle initial state: if last_kick_ms is 0 and tick_ms is 0, no timeout */
    if (wdt->last_kick_ms == 0 && tick_ms == 0) {
        return false;
    }

    bool expired = (elapsed > wdt->timeout_ms);
    if (expired && !wdt->triggered) {
        wdt->triggered = true;
        wdt->trigger_count++;
    }
    return expired;
}

void watchdog_disable(watchdog_t *wdt) {
    if (!wdt) return;
    wdt->enabled = false;
}
