/**
 * @file safety_sm.h
 * @brief NEXUS Safety System — Event-driven safety state machine.
 *
 * States: NORMAL -> DEGRADED -> SAFE_STATE -> FAULT
 * The system does NOT auto-resume. Explicit RESUME required.
 *
 * Events drive all state transitions. No automatic heartbeat-based
 * transitions (that logic lives in hal.c).
 */

#ifndef NEXUS_SAFETY_SM_H
#define NEXUS_SAFETY_SM_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Safety state — use guard to avoid re-definition with hal.h */
#ifndef NEXUS_SAFETY_STATE_DEFINED
#define NEXUS_SAFETY_STATE_DEFINED
typedef enum {
    SAFETY_NORMAL = 0,
    SAFETY_DEGRADED = 1,
    SAFETY_SAFE_STATE = 2,
    SAFETY_FAULT = 3,
} safety_state_t;
#endif

/* Safety event types */
typedef enum {
    SAFETY_EVT_HEARTBEAT_MISS,
    SAFETY_EVT_HEARTBEAT_RECOVER,
    SAFETY_EVT_ESTOP_TRIGGERED,
    SAFETY_EVT_OVERCURRENT,
    SAFETY_EVT_WATCHDOG_TIMEOUT,
    SAFETY_EVT_SENSOR_FAILURE,
    SAFETY_EVT_TRUST_VIOLATION,
    SAFETY_EVT_RESUME_COMMAND,
    SAFETY_EVT_BOOT_COMPLETE,
} safety_event_t;

typedef struct {
    safety_state_t current_state;
    uint32_t state_entry_time_ms;
    uint32_t boot_count;
    uint32_t boot_times_ms[5];  /* Last 5 boot times */
    safety_event_t last_event;
    uint32_t last_event_time_ms;
} safety_sm_t;

/**
 * @brief Process a safety event and return new state.
 */
safety_state_t safety_sm_process_event(safety_sm_t *sm, safety_event_t event, uint32_t tick_ms);

/**
 * @brief Get human-readable state name.
 */
const char* safety_sm_state_name(safety_state_t state);

/**
 * @brief Initialize safety state machine to NORMAL.
 */
void safety_sm_init(safety_sm_t *sm);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_SAFETY_SM_H */
