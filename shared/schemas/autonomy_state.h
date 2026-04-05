/**
 * @file autonomy_state.h
 * @brief NEXUS Autonomy State — C struct layouts matching autonomy_state.json schema.
 *
 * Structs for trust scores, autonomy levels, events, and INCREMENTS parameters.
 * Field names and types match the JSON schema definitions exactly.
 */

#ifndef NEXUS_AUTONOMY_STATE_H
#define NEXUS_AUTONOMY_STATE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Trust Parameters (INCREMENTS algorithm, 12 fields)
 * Matches: autonomy_state.json#/definitions/trust_params
 * =================================================================== */

typedef struct {
    float alpha_gain;                /* Gain rate per evaluation window (default 0.002) */
    float alpha_loss;                /* Loss multiplier, 25:1 vs alpha_gain (default 0.05) */
    float alpha_decay;               /* Idle decay rate toward floor (default 0.0001) */
    float t_floor;                   /* Trust floor for penalty/decay (default 0.2) */
    int   quality_cap;               /* Max good events per window (default 10) */
    float evaluation_window_hours;   /* Window duration in hours (default 1.0) */
    float severity_exponent;         /* Severity exponent in penalty (default 1.0) */
    float streak_bonus;              /* Per-clean-window bonus (default 0.00005) */
    int   min_events_for_gain;       /* Min good events for gain branch (default 1) */
    float n_penalty_slope;           /* Multi-bad-event amplification (default 0.1) */
    float reset_grace_hours;         /* Hours between resets (default 24.0) */
    float promotion_cooldown_hours;  /* Min hours between promotions (default 72.0) */
} nexus_trust_params_t;

/* ===================================================================
 * Trust Event (15 defined types)
 * Matches: autonomy_state.json#/definitions/trust_event
 * =================================================================== */

typedef struct {
    char   event_type[32];     /* Event type string (e.g. "heartbeat_ok") */
    float  quality;            /* Quality metric [0.0, 1.0] (for good events) */
    float  severity;           /* Severity metric [0.0, 1.0] (for bad events) */
    double timestamp;          /* Unix timestamp (seconds) */
    char   subsystem[24];      /* Subsystem name (e.g. "steering") */
    bool   is_bad;             /* Whether this is a bad event */
} nexus_trust_event_t;

/* ===================================================================
 * Subsystem Trust State
 * Matches: autonomy_state.json#/definitions/subsystem_trust
 * =================================================================== */

#define NEXUS_MAX_SUBSYSTEMS       5
#define NEXUS_MAX_SEVERITY_HISTORY 500

typedef struct {
    char   subsystem[24];                /* Subsystem name */
    float  trust_score;                  /* Current trust [0.0, 1.0] */
    int    autonomy_level;               /* Current level (L0=0 to L5=5) */
    int    consecutive_clean_windows;    /* Windows since last bad event */
    int    total_windows;                /* Total evaluation windows */
    int    clean_windows;                /* Total clean windows */
    float  total_observation_hours;      /* Cumulative observation time */
    double last_promotion_time;          /* Timestamp of last promotion */
    double last_reset_time;              /* Timestamp of last reset */
} nexus_subsystem_trust_t;

/* ===================================================================
 * Autonomy Level Definition
 * Matches: autonomy_state.json#/definitions/autonomy_level_def
 * =================================================================== */

#define NEXUS_NUM_AUTONOMY_LEVELS 6

typedef struct {
    int    level;                     /* Level number (0-5) */
    char   name[24];                  /* Level name (e.g. "Semi-Autonomous") */
    float  trust_threshold;           /* Required trust (NAN for L0) */
    float  min_observation_hours;     /* Minimum observation hours */
    int    min_clean_windows;         /* Minimum clean windows */
    char   key_criteria[64];          /* Human-readable criteria */
    bool   has_threshold;             /* Whether trust_threshold is valid */
} nexus_autonomy_level_def_t;

/* ===================================================================
 * Trust Update Result
 * Matches: autonomy_state.json#/definitions/trust_update_result
 * =================================================================== */

typedef struct {
    char   subsystem[24];     /* Subsystem name */
    float  old_score;         /* Score before evaluation */
    float  new_score;         /* Score after evaluation */
    float  delta;             /* Change in trust score */
    char   branch[8];         /* "gain", "penalty", or "decay" */
    int    n_good;            /* Number of good events */
    int    n_bad;             /* Number of bad events */
    float  max_severity;      /* Maximum severity among bad events */
    int    old_level;         /* Autonomy level before evaluation */
    int    new_level;         /* Autonomy level after evaluation */
    char   reason[128];       /* Human-readable reason */
} nexus_trust_update_result_t;

/* ===================================================================
 * Top-level Autonomy State
 * Matches: autonomy_state.json (root object)
 * =================================================================== */

typedef struct {
    nexus_trust_params_t        params;
    int                         num_subsystems;
    nexus_subsystem_trust_t     subsystems[NEXUS_MAX_SUBSYSTEMS];
    nexus_autonomy_level_def_t  autonomy_levels[NEXUS_NUM_AUTONOMY_LEVELS];
} nexus_autonomy_state_t;

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_AUTONOMY_STATE_H */
