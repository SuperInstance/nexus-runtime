/**
 * @file safety_sm.h
 * @brief NEXUS Safety System - Safety state machine.
 *
 * States: NORMAL -> DEGRADED -> SAFE_STATE -> FAULT
 * The system does NOT auto-resume. Explicit RESUME required.
 */

#ifndef NEXUS_SAFETY_SM_H
#define NEXUS_SAFETY_SM_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SAFETY_NORMAL = 0,
    SAFETY_DEGRADED = 1,
    SAFETY_SAFE_STATE = 2,
    SAFETY_FAULT = 3,
} safety_state_t;

typedef struct {
    safety_state_t state;
    uint32_t missed_heartbeats;
    uint32_t last_heartbeat_ms;
    uint32_t state_entry_ms;
    uint32_t fault_count;
} safety_sm_t;

/**
 * @brief Initialize safety state machine.
 * @param sm Pointer to state machine.
 */
void safety_sm_init(safety_sm_t *sm);

/**
 * @brief Update safety state machine (call periodically).
 * @param sm Pointer to state machine.
 * @param now_ms Current time in milliseconds.
 * @return Current safety state.
 */
safety_state_t safety_sm_update(safety_sm_t *sm, uint32_t now_ms);

/**
 * @brief Record a heartbeat received.
 * @param sm Pointer to state machine.
 * @param now_ms Current time in milliseconds.
 */
void safety_sm_heartbeat(safety_sm_t *sm, uint32_t now_ms);

/**
 * @brief Request transition to safe state.
 * @param sm Pointer to state machine.
 * @param now_ms Current time in milliseconds.
 */
void safety_sm_request_safe_state(safety_sm_t *sm, uint32_t now_ms);

/**
 * @brief Explicit resume from safe state.
 * @param sm Pointer to state machine.
 * @param now_ms Current time in milliseconds.
 * @return true if resume was allowed, false otherwise.
 */
bool safety_sm_resume(safety_sm_t *sm, uint32_t now_ms);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_SAFETY_SM_H */
