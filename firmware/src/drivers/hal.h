/**
 * @file hal.h
 * @brief NEXUS Hardware Abstraction Layer — sensor I/O, actuator control, safety.
 *
 * The HAL sits between the VM and actual ESP32 hardware.
 * On the host (without ESP-IDF), stubs simulate hardware for testing.
 *
 * Provides: 64 sensor channels, 64 actuator channels with safety profiles,
 * four-tier safety state machine, watchdog, heartbeat monitor,
 * overcurrent detection, and E-Stop handling.
 */

#ifndef NEXUS_HAL_H
#define NEXUS_HAL_H

#include <stdint.h>
#include <stdbool.h>

/* ===================================================================
 * HAL error codes
 * =================================================================== */

typedef enum {
    HAL_OK = 0,
    HAL_ERR_NOT_READY,
    HAL_ERR_TIMEOUT,
    HAL_ERR_INVALID_PIN,
    HAL_ERR_OVERCURRENT,
    HAL_ERR_COMM_FAILED,
} hal_error_t;

/* ===================================================================
 * Pin modes
 * =================================================================== */

typedef enum {
    PIN_MODE_INPUT,
    PIN_MODE_OUTPUT,
    PIN_MODE_INPUT_PULLUP,
    PIN_MODE_INPUT_PULLDOWN,
} pin_mode_t;

/* ===================================================================
 * Actuator types
 * =================================================================== */

typedef enum {
    ACTUATOR_SERVO,
    ACTUATOR_RELAY,
    ACTUATOR_MOTOR_PWM,
    ACTUATOR_SOLENOID,
    ACTUATOR_LED,
    ACTUATOR_BUZZER,
} actuator_type_t;

/* ===================================================================
 * Safety states (guard against re-definition)
 * =================================================================== */

#ifndef NEXUS_SAFETY_STATE_DEFINED
#define NEXUS_SAFETY_STATE_DEFINED
typedef enum {
    SAFETY_NORMAL,
    SAFETY_DEGRADED,
    SAFETY_SAFE_STATE,
    SAFETY_FAULT,
} safety_state_t;
#endif

/* ===================================================================
 * Sensor reading
 * =================================================================== */

typedef struct {
    uint8_t  sensor_id;
    float    value;
    bool     valid;
    uint32_t timestamp_ms;
} sensor_reading_t;

/* ===================================================================
 * Actuator command
 * =================================================================== */

typedef struct {
    uint8_t  actuator_id;
    float    value;
    bool     enabled;
    actuator_type_t type;
} actuator_command_t;

/* ===================================================================
 * Actuator safety profile
 * =================================================================== */

typedef struct {
    float    safe_value;
    float    min_value;
    float    max_value;
    float    max_rate_per_tick;  /* Max change per 1ms tick */
    float    overcurrent_limit_ma;
    float    prev_value;
    bool     is_active;
} actuator_profile_t;

/* ===================================================================
 * HAL device context
 * =================================================================== */

typedef struct {
    /* Sensor registers - populated before each VM tick */
    sensor_reading_t sensors[64];

    /* Actuator registers - drained after each VM tick */
    actuator_command_t actuators[64];
    actuator_profile_t actuator_profiles[64];

    /* Safety state */
    safety_state_t safety_state;
    bool estop_triggered;
    uint32_t estop_trigger_time_ms;

    /* Watchdog */
    bool hw_wdt_enabled;
    uint32_t hw_wdt_last_feed_ms;
    uint32_t hw_wdt_timeout_ms;
    uint8_t  hw_wdt_kick_pattern;  /* Alternating 0x55/0xAA */

    /* Heartbeat */
    uint32_t last_heartbeat_ms;
    uint32_t heartbeat_miss_count;
    uint32_t heartbeat_degrade_threshold;  /* 5 misses */
    uint32_t heartbeat_safe_threshold;      /* 10 misses */

    /* System timing */
    uint32_t tick_count_ms;
    uint32_t uptime_ms;

    /* Overcurrent monitoring */
    float current_ma[64];
    float overcurrent_threshold_ma[64];
    bool  overcurrent_triggered[64];
} hal_context_t;

/* ===================================================================
 * HAL API
 * =================================================================== */

hal_error_t hal_init(hal_context_t *ctx);
hal_error_t hal_configure_sensor(hal_context_t *ctx, uint8_t id, pin_mode_t mode);
hal_error_t hal_configure_actuator(hal_context_t *ctx, uint8_t id, actuator_type_t type,
                                     float safe_value, float min_val, float max_val,
                                     float max_rate, float oc_threshold);
hal_error_t hal_read_sensor(hal_context_t *ctx, uint8_t id, float *value);
hal_error_t hal_write_actuator(hal_context_t *ctx, uint8_t id, float value);
hal_error_t hal_update_sensors(hal_context_t *ctx, uint32_t tick_ms);
hal_error_t hal_drain_actuators(hal_context_t *ctx);
hal_error_t hal_apply_actuator_limits(hal_context_t *ctx);
void hal_trigger_estop(hal_context_t *ctx);
void hal_safe_all_actuators(hal_context_t *ctx);
hal_error_t hal_feed_watchdog(hal_context_t *ctx);
bool hal_check_watchdog(hal_context_t *ctx);
safety_state_t hal_update_safety_state(hal_context_t *ctx, uint32_t tick_ms);
void hal_record_heartbeat(hal_context_t *ctx, uint32_t tick_ms);
bool hal_check_overcurrent(hal_context_t *ctx, uint8_t actuator_id);
const char* safety_state_name(safety_state_t state);

#endif /* NEXUS_HAL_H */
