/**
 * @file safety_sm.c
 * @brief NEXUS Safety System — Event-driven safety state machine implementation.
 *
 * Four-tier state machine: NORMAL -> DEGRADED -> SAFE_STATE -> FAULT
 * No auto-resume. Explicit SAFETY_EVT_RESUME_COMMAND required to go
 * from SAFE_STATE back to NORMAL.
 */

#include "safety_sm.h"
#include <string.h>

/* Ten minutes in milliseconds — used for FAULT boot counter window */
#define FAULT_BOOT_WINDOW_MS  (10UL * 60UL * 1000UL)
#define FAULT_BOOT_THRESHOLD  5

void safety_sm_init(safety_sm_t *sm) {
    if (!sm) return;
    memset(sm, 0, sizeof(safety_sm_t));
    sm->current_state = SAFETY_NORMAL;
}

safety_state_t safety_sm_process_event(safety_sm_t *sm, safety_event_t event, uint32_t tick_ms) {
    if (!sm) return SAFETY_FAULT;

    sm->last_event = event;
    sm->last_event_time_ms = tick_ms;

    switch (sm->current_state) {

    case SAFETY_NORMAL:
        switch (event) {
        case SAFETY_EVT_HEARTBEAT_MISS:
            /* Note: The HAL handles threshold counting. This just acknowledges. */
            break;
        case SAFETY_EVT_ESTOP_TRIGGERED:
        case SAFETY_EVT_OVERCURRENT:
        case SAFETY_EVT_WATCHDOG_TIMEOUT:
        case SAFETY_EVT_SENSOR_FAILURE:
        case SAFETY_EVT_TRUST_VIOLATION:
            sm->current_state = SAFETY_DEGRADED;
            sm->state_entry_time_ms = tick_ms;
            break;
        case SAFETY_EVT_HEARTBEAT_RECOVER:
        case SAFETY_EVT_BOOT_COMPLETE:
            break;
        case SAFETY_EVT_RESUME_COMMAND:
            /* Already in NORMAL, nothing to do */
            break;
        }
        break;

    case SAFETY_DEGRADED:
        switch (event) {
        case SAFETY_EVT_HEARTBEAT_MISS:
        case SAFETY_EVT_OVERCURRENT:
        case SAFETY_EVT_WATCHDOG_TIMEOUT:
        case SAFETY_EVT_SENSOR_FAILURE:
        case SAFETY_EVT_TRUST_VIOLATION:
        case SAFETY_EVT_ESTOP_TRIGGERED:
            sm->current_state = SAFETY_SAFE_STATE;
            sm->state_entry_time_ms = tick_ms;
            break;
        case SAFETY_EVT_HEARTBEAT_RECOVER:
            /* Recover back to NORMAL from DEGRADED */
            sm->current_state = SAFETY_NORMAL;
            sm->state_entry_time_ms = tick_ms;
            break;
        case SAFETY_EVT_RESUME_COMMAND:
            /* Resume from DEGRADED to NORMAL */
            sm->current_state = SAFETY_NORMAL;
            sm->state_entry_time_ms = tick_ms;
            break;
        case SAFETY_EVT_BOOT_COMPLETE:
            break;
        }
        break;

    case SAFETY_SAFE_STATE:
        switch (event) {
        case SAFETY_EVT_ESTOP_TRIGGERED:
        case SAFETY_EVT_WATCHDOG_TIMEOUT:
        case SAFETY_EVT_SENSOR_FAILURE:
        case SAFETY_EVT_TRUST_VIOLATION:
            /* Check boot counter for FAULT transition */
            {
                /* Count boots within the last 10 minutes */
                uint32_t recent_boots = 0;
                for (uint32_t i = 0; i < 5 && i < sm->boot_count; i++) {
                    if (tick_ms - sm->boot_times_ms[i] < FAULT_BOOT_WINDOW_MS) {
                        recent_boots++;
                    }
                }

                if (recent_boots > FAULT_BOOT_THRESHOLD) {
                    sm->current_state = SAFETY_FAULT;
                    sm->state_entry_time_ms = tick_ms;
                    break;
                }
            }
            /* Stay in SAFE_STATE */
            break;
        case SAFETY_EVT_RESUME_COMMAND:
            /* Explicit resume from SAFE_STATE back to NORMAL */
            sm->current_state = SAFETY_NORMAL;
            sm->state_entry_time_ms = tick_ms;
            break;
        case SAFETY_EVT_HEARTBEAT_MISS:
        case SAFETY_EVT_HEARTBEAT_RECOVER:
        case SAFETY_EVT_OVERCURRENT:
            break;
        case SAFETY_EVT_BOOT_COMPLETE:
            /* Record boot time for FAULT boot-rate detection */
            sm->boot_times_ms[sm->boot_count % 5] = tick_ms;
            sm->boot_count++;
            break;
        }
        break;

    case SAFETY_FAULT:
        /* Terminal state — no transitions out */
        break;
    }

    return sm->current_state;
}

const char* safety_sm_state_name(safety_state_t state) {
    switch (state) {
    case SAFETY_NORMAL:     return "NORMAL";
    case SAFETY_DEGRADED:   return "DEGRADED";
    case SAFETY_SAFE_STATE: return "SAFE_STATE";
    case SAFETY_FAULT:      return "FAULT";
    default:                return "UNKNOWN";
    }
}
