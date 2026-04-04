/**
 * @file safety_sm.c
 * @brief NEXUS Safety System - Safety state machine implementation (stub).
 */

#include "safety_sm.h"
#include <string.h>

void safety_sm_init(safety_sm_t *sm) {
    if (!sm) {
        return;
    }
    memset(sm, 0, sizeof(safety_sm_t));
    sm->state = SAFETY_NORMAL;
    sm->missed_heartbeats = 0;
}

safety_state_t safety_sm_update(safety_sm_t *sm, uint32_t now_ms) {
    if (!sm) {
        return SAFETY_FAULT;
    }

    (void)now_ms;

    /* TODO: Implement full state machine transitions */
    /* NORMAL -> DEGRADED: 5 missed heartbeats (500ms) */
    /* DEGRADED -> SAFE_STATE: 10 missed heartbeats (1000ms) */
    /* SAFE_STATE -> FAULT: unrecoverable error */

    return sm->state;
}

void safety_sm_heartbeat(safety_sm_t *sm, uint32_t now_ms) {
    if (!sm) {
        return;
    }
    sm->missed_heartbeats = 0;
    sm->last_heartbeat_ms = now_ms;
}

void safety_sm_request_safe_state(safety_sm_t *sm, uint32_t now_ms) {
    if (!sm) {
        return;
    }
    if (sm->state < SAFETY_SAFE_STATE) {
        sm->state = SAFETY_SAFE_STATE;
        sm->state_entry_ms = now_ms;
    }
}

bool safety_sm_resume(safety_sm_t *sm, uint32_t now_ms) {
    if (!sm) {
        return false;
    }
    if (sm->state == SAFETY_FAULT) {
        return false;
    }
    if (sm->state == SAFETY_SAFE_STATE) {
        sm->state = SAFETY_NORMAL;
        sm->state_entry_ms = now_ms;
        sm->missed_heartbeats = 0;
        return true;
    }
    return false;
}
