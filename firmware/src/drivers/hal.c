/**
 * @file hal.c
 * @brief NEXUS Hardware Abstraction Layer — implementation.
 *
 * Provides sensor I/O, actuator control with safety profiles,
 * rate limiting, NaN/Infinity guards, E-Stop, watchdog, heartbeat,
 * and overcurrent detection.
 *
 * On host (without ESP-IDF), all hardware access is simulated with
 * deterministic stubs for testing.
 */

#include "hal.h"
#include <string.h>
#include <math.h>

/* ===================================================================
 * Internal helpers
 * =================================================================== */

/** Check if float bits represent NaN or Infinity. */
static inline bool hal_is_nan_or_inf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (bits & 0x7F800000u) == 0x7F800000u;
}

/** Get sign of a float: +1.0f or -1.0f. */
static inline float hal_sign(float f) {
    return (f >= 0.0f) ? 1.0f : -1.0f;
}

/* ===================================================================
 * hal_init — Zero all state, set defaults
 * =================================================================== */

hal_error_t hal_init(hal_context_t *ctx) {
    if (!ctx) return HAL_ERR_INVALID_PIN;

    memset(ctx, 0, sizeof(hal_context_t));

    /* Default heartbeat thresholds */
    ctx->heartbeat_degrade_threshold = 5;
    ctx->heartbeat_safe_threshold = 10;

    /* Enable watchdog with 1000ms timeout */
    ctx->hw_wdt_enabled = true;
    ctx->hw_wdt_timeout_ms = 1000;
    ctx->hw_wdt_last_feed_ms = 0;
    ctx->hw_wdt_kick_pattern = 0x55; /* First kick expects 0x55 */

    /* Safety state starts at NORMAL */
    ctx->safety_state = SAFETY_NORMAL;
    ctx->estop_triggered = false;

    /* Overcurrent defaults: no thresholds set, no triggers */
    for (int i = 0; i < 64; i++) {
        ctx->overcurrent_threshold_ma[i] = 0.0f;
        ctx->current_ma[i] = 0.0f;
        ctx->overcurrent_triggered[i] = false;
    }

    return HAL_OK;
}

/* ===================================================================
 * hal_configure_sensor — Set sensor pin mode, mark as valid
 * =================================================================== */

hal_error_t hal_configure_sensor(hal_context_t *ctx, uint8_t id, pin_mode_t mode) {
    if (!ctx || id >= 64) return HAL_ERR_INVALID_PIN;
    (void)mode; /* Mode stored for ESP32; stub ignores it */
    ctx->sensors[id].sensor_id = id;
    ctx->sensors[id].valid = true;
    ctx->sensors[id].value = 0.0f;
    ctx->sensors[id].timestamp_ms = ctx->tick_count_ms;
    return HAL_OK;
}

/* ===================================================================
 * hal_configure_actuator — Set actuator type and safety profile
 * =================================================================== */

hal_error_t hal_configure_actuator(hal_context_t *ctx, uint8_t id, actuator_type_t type,
                                     float safe_value, float min_val, float max_val,
                                     float max_rate, float oc_threshold) {
    if (!ctx || id >= 64) return HAL_ERR_INVALID_PIN;

    ctx->actuator_profiles[id].safe_value = safe_value;
    ctx->actuator_profiles[id].min_value = min_val;
    ctx->actuator_profiles[id].max_value = max_val;
    ctx->actuator_profiles[id].max_rate_per_tick = max_rate;
    ctx->actuator_profiles[id].overcurrent_limit_ma = oc_threshold;
    ctx->actuator_profiles[id].prev_value = safe_value;
    ctx->actuator_profiles[id].is_active = true;

    ctx->actuators[id].type = type;
    ctx->actuators[id].actuator_id = id;
    ctx->actuators[id].enabled = true;
    ctx->actuators[id].value = safe_value;

    ctx->overcurrent_threshold_ma[id] = oc_threshold;
    ctx->current_ma[id] = 0.0f;
    ctx->overcurrent_triggered[id] = false;

    return HAL_OK;
}

/* ===================================================================
 * hal_read_sensor — Return sensor value
 * =================================================================== */

hal_error_t hal_read_sensor(hal_context_t *ctx, uint8_t id, float *value) {
    if (!ctx || !value) return HAL_ERR_INVALID_PIN;
    if (id >= 64) return HAL_ERR_INVALID_PIN;
    if (!ctx->sensors[id].valid) return HAL_ERR_NOT_READY;

    *value = ctx->sensors[id].value;
    return HAL_OK;
}

/* ===================================================================
 * hal_write_actuator — Store actuator command (no limits yet)
 * =================================================================== */

hal_error_t hal_write_actuator(hal_context_t *ctx, uint8_t id, float value) {
    if (!ctx) return HAL_ERR_INVALID_PIN;
    if (id >= 64) return HAL_ERR_INVALID_PIN;

    ctx->actuators[id].value = value;
    ctx->actuators[id].enabled = true;
    return HAL_OK;
}

/* ===================================================================
 * hal_update_sensors — Simulate sensor reading (host stub)
 * =================================================================== */

hal_error_t hal_update_sensors(hal_context_t *ctx, uint32_t tick_ms) {
    if (!ctx) return HAL_ERR_INVALID_PIN;

    ctx->tick_count_ms = tick_ms;
    ctx->uptime_ms = tick_ms;

    /* On host: generate deterministic simulated data for configured sensors.
     * Each sensor produces a sine wave based on sensor_id and tick_ms. */
    for (int i = 0; i < 64; i++) {
        if (ctx->sensors[i].valid) {
            /* Deterministic simulation: value = sensor_id + sin(tick/1000) */
            float phase = (float)i * 0.5f;
            float t = (float)tick_ms * 0.001f;
            ctx->sensors[i].value = phase + sinf(t + phase) * 10.0f;
            ctx->sensors[i].timestamp_ms = tick_ms;
        }
    }

    return HAL_OK;
}

/* ===================================================================
 * hal_drain_actuators — Process actuator commands after VM tick
 * =================================================================== */

hal_error_t hal_drain_actuators(hal_context_t *ctx) {
    if (!ctx) return HAL_ERR_INVALID_PIN;

    /* Apply safety limits to all active actuator commands */
    hal_apply_actuator_limits(ctx);

    /* Check overcurrent on all channels */
    for (int i = 0; i < 64; i++) {
        if (ctx->actuator_profiles[i].is_active) {
            hal_check_overcurrent(ctx, (uint8_t)i);
        }
    }

    return HAL_OK;
}

/* ===================================================================
 * hal_apply_actuator_limits — Safety checks and rate limiting
 *
 * For each actuator with a pending command:
 *   1. Check if estop is triggered → force safe_value
 *   2. Check if safety_state != NORMAL → force safe_value
 *   3. Rate limiting: delta = value - prev_value;
 *      if abs(delta) > max_rate: value = prev_value + sign(delta) * max_rate
 *   4. Clamp to [min_value, max_value]
 *   5. NaN/Infinity guard: if isnormal(value) is false, use safe_value
 *   6. Update prev_value
 * =================================================================== */

hal_error_t hal_apply_actuator_limits(hal_context_t *ctx) {
    if (!ctx) return HAL_ERR_INVALID_PIN;

    for (int i = 0; i < 64; i++) {
        actuator_profile_t *prof = &ctx->actuator_profiles[i];
        if (!prof->is_active) continue;

        actuator_command_t *cmd = &ctx->actuators[i];
        if (!cmd->enabled) continue;

        float value = cmd->value;

        /* Step 1: E-Stop check → force safe value */
        if (ctx->estop_triggered) {
            value = prof->safe_value;
            goto apply;
        }

        /* Step 2: Safety state check → force safe value if not NORMAL */
        if (ctx->safety_state != SAFETY_NORMAL) {
            value = prof->safe_value;
            goto apply;
        }

        /* Step 5: NaN/Infinity guard (before rate limiting to avoid
         * corrupting prev_value) */
        if (!isnormal(value)) {
            value = prof->safe_value;
            goto apply;
        }

        /* Step 3: Rate limiting */
        if (prof->max_rate_per_tick > 0.0f) {
            float delta = value - prof->prev_value;
            if (fabsf(delta) > prof->max_rate_per_tick) {
                value = prof->prev_value + hal_sign(delta) * prof->max_rate_per_tick;
            }
        }

        /* Step 4: Clamp to [min_value, max_value] */
        if (value < prof->min_value) value = prof->min_value;
        if (value > prof->max_value) value = prof->max_value;

apply:
        /* Update prev_value and store result */
        prof->prev_value = value;
        cmd->value = value;
    }

    return HAL_OK;
}

/* ===================================================================
 * hal_trigger_estop — Emergency stop: safe all actuators
 * =================================================================== */

void hal_trigger_estop(hal_context_t *ctx) {
    if (!ctx) return;

    ctx->estop_triggered = true;
    ctx->estop_trigger_time_ms = ctx->tick_count_ms;

    /* Force all actuators to safe state */
    hal_safe_all_actuators(ctx);
}

/* ===================================================================
 * hal_safe_all_actuators — Set all actuators to safe value
 * =================================================================== */

void hal_safe_all_actuators(hal_context_t *ctx) {
    if (!ctx) return;

    for (int i = 0; i < 64; i++) {
        if (ctx->actuator_profiles[i].is_active) {
            ctx->actuators[i].value = ctx->actuator_profiles[i].safe_value;
            ctx->actuator_profiles[i].prev_value = ctx->actuator_profiles[i].safe_value;
        }
    }
}

/* ===================================================================
 * hal_feed_watchdog — Update last feed time with alternating pattern
 * =================================================================== */

hal_error_t hal_feed_watchdog(hal_context_t *ctx) {
    if (!ctx) return HAL_ERR_INVALID_PIN;

    /* Alternating 0x55/0xAA pattern */
    if (ctx->hw_wdt_kick_pattern == 0x55) {
        ctx->hw_wdt_kick_pattern = 0xAA;
    } else {
        ctx->hw_wdt_kick_pattern = 0x55;
    }

    ctx->hw_wdt_last_feed_ms = ctx->tick_count_ms;
    return HAL_OK;
}

/* ===================================================================
 * hal_check_watchdog — Return true if watchdog has expired
 * =================================================================== */

bool hal_check_watchdog(hal_context_t *ctx) {
    if (!ctx || !ctx->hw_wdt_enabled) return false;

    uint32_t elapsed = ctx->tick_count_ms - ctx->hw_wdt_last_feed_ms;
    /* Handle wraparound: if tick_count_ms < last_feed_ms after boot,
     * last_feed_ms starts at 0 so elapsed = tick_count_ms */
    if (ctx->tick_count_ms < ctx->hw_wdt_last_feed_ms) {
        elapsed = ctx->tick_count_ms; /* Just started */
    }

    return elapsed > ctx->hw_wdt_timeout_ms;
}

/* ===================================================================
 * hal_update_safety_state — Check heartbeat misses and update state
 *
 * State transitions:
 *   NORMAL → DEGRADED: >= 5 heartbeats missed
 *   DEGRADED → SAFE_STATE: >= 10 heartbeats missed
 *   SAFE_STATE → FAULT: boot counter > 5 in 10 minutes
 *   No auto-resume from any state
 * =================================================================== */

safety_state_t hal_update_safety_state(hal_context_t *ctx, uint32_t tick_ms) {
    if (!ctx) return SAFETY_FAULT;

    ctx->tick_count_ms = tick_ms;

    /* Only progress through degraded states based on heartbeat misses.
     * E-Stop directly forces SAFE_STATE. */
    if (ctx->estop_triggered && ctx->safety_state < SAFETY_SAFE_STATE) {
        ctx->safety_state = SAFETY_SAFE_STATE;
        return ctx->safety_state;
    }

    /* Check heartbeat misses for state degradation */
    switch (ctx->safety_state) {
    case SAFETY_NORMAL:
        if (ctx->heartbeat_miss_count >= ctx->heartbeat_degrade_threshold) {
            ctx->safety_state = SAFETY_DEGRADED;
        }
        break;

    case SAFETY_DEGRADED:
        if (ctx->heartbeat_miss_count >= ctx->heartbeat_safe_threshold) {
            ctx->safety_state = SAFETY_SAFE_STATE;
        }
        break;

    case SAFETY_SAFE_STATE:
        /* Could transition to FAULT based on boot counter,
         * but that requires external tracking. Stay in SAFE_STATE. */
        break;

    case SAFETY_FAULT:
        /* Terminal state. No transitions out. */
        break;
    }

    return ctx->safety_state;
}

/* ===================================================================
 * hal_record_heartbeat — Reset miss counter, update last heartbeat
 * =================================================================== */

void hal_record_heartbeat(hal_context_t *ctx, uint32_t tick_ms) {
    if (!ctx) return;

    ctx->heartbeat_miss_count = 0;
    ctx->last_heartbeat_ms = tick_ms;
}

/* ===================================================================
 * hal_check_overcurrent — Check if actuator exceeded current threshold
 * =================================================================== */

bool hal_check_overcurrent(hal_context_t *ctx, uint8_t actuator_id) {
    if (!ctx || actuator_id >= 64) return false;

    if (ctx->overcurrent_threshold_ma[actuator_id] > 0.0f &&
        ctx->current_ma[actuator_id] > ctx->overcurrent_threshold_ma[actuator_id]) {
        ctx->overcurrent_triggered[actuator_id] = true;
        return true;
    }

    ctx->overcurrent_triggered[actuator_id] = false;
    return false;
}

/* ===================================================================
 * safety_state_name — Return string name for safety state
 * =================================================================== */

const char* safety_state_name(safety_state_t state) {
    switch (state) {
    case SAFETY_NORMAL:     return "NORMAL";
    case SAFETY_DEGRADED:   return "DEGRADED";
    case SAFETY_SAFE_STATE: return "SAFE_STATE";
    case SAFETY_FAULT:      return "FAULT";
    default:                return "UNKNOWN";
    }
}
