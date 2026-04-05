/**
 * @file node_role_config.h
 * @brief NEXUS Node Role Config — C struct layouts matching node_role_config.json schema.
 *
 * Structs for pin configs, safety configs, reflex bindings, and serial config.
 * Field names and types match the JSON schema definitions exactly.
 */

#ifndef NEXUS_NODE_ROLE_CONFIG_H
#define NEXUS_NODE_ROLE_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Constants
 * =================================================================== */

#define NEXUS_NODE_ID_MAX      32
#define NEXUS_ROLE_NAME_MAX    32
#define NEXUS_PIN_LABEL_MAX    32
#define NEXUS_SENSOR_TYPE_MAX  32
#define NEXUS_REFLEX_BIND_MAX  8
#define NEXUS_MAX_CHANNELS     64
#define NEXUS_VERSION_MAX      16

/* ===================================================================
 * Pin Mode
 * Matches: node_role_config.json#/definitions/pin_config/properties/mode
 * =================================================================== */

typedef enum {
    NEXUS_PIN_MODE_INPUT         = 0,
    NEXUS_PIN_MODE_OUTPUT        = 1,
    NEXUS_PIN_MODE_INPUT_PULLUP  = 2,
    NEXUS_PIN_MODE_INPUT_PULLDOWN = 3,
} nexus_pin_mode_t;

/* ===================================================================
 * Actuator Type
 * Matches: node_role_config.json#/definitions/actuator_pin_config/.../actuator_type
 * =================================================================== */

typedef enum {
    NEXUS_ACTUATOR_SERVO      = 0,
    NEXUS_ACTUATOR_RELAY      = 1,
    NEXUS_ACTUATOR_MOTOR_PWM  = 2,
    NEXUS_ACTUATOR_SOLENOID   = 3,
    NEXUS_ACTUATOR_LED        = 4,
    NEXUS_ACTUATOR_BUZZER     = 5,
} nexus_actuator_type_t;

/* ===================================================================
 * Pin Config (base)
 * Matches: node_role_config.json#/definitions/pin_config
 * =================================================================== */

typedef struct {
    int    id;                  /* Pin/channel ID (0-63) */
    nexus_pin_mode_t mode;      /* GPIO mode */
    char   label[NEXUS_PIN_LABEL_MAX]; /* Human-readable label */
} nexus_pin_config_t;

/* ===================================================================
 * Sensor Pin Config
 * Matches: node_role_config.json#/definitions/sensor_pin_config
 * =================================================================== */

typedef struct {
    nexus_pin_config_t base;    /* Base pin config */
    char sensor_type[NEXUS_SENSOR_TYPE_MAX]; /* Sensor type identifier */
} nexus_sensor_pin_config_t;

/* ===================================================================
 * Actuator Pin Config
 * Matches: node_role_config.json#/definitions/actuator_pin_config
 * =================================================================== */

typedef struct {
    nexus_pin_config_t   base;               /* Base pin config */
    nexus_actuator_type_t actuator_type;      /* Actuator hardware type */
    float                safe_value;          /* Safe fallback value */
    float                min_value;           /* Minimum allowed value */
    float                max_value;           /* Maximum allowed value */
    float                max_rate_per_tick;   /* Max change per 1ms tick */
    float                overcurrent_limit_ma; /* Overcurrent threshold */
    bool                 has_limits;          /* Whether min/max_value are set */
    bool                 has_rate;            /* Whether max_rate_per_tick is set */
    bool                 has_oc_limit;        /* Whether overcurrent_limit_ma is set */
} nexus_actuator_pin_config_t;

/* ===================================================================
 * Safety Config
 * Matches: node_role_config.json#/definitions/safety_config
 * =================================================================== */

typedef struct {
    int   heartbeat_degrade_threshold;  /* Missed HBs before DEGRADED (default 5) */
    int   heartbeat_safe_threshold;     /* Missed HBs before SAFE_STATE (default 10) */
    int   watchdog_timeout_ms;          /* Watchdog timeout (default 3000) */
    int   estop_pin;                    /* E-Stop pin (-1 if unused) */
    int   overcurrent_pins[NEXUS_MAX_CHANNELS]; /* OC monitor pins */
    int   num_overcurrent_pins;         /* Count of OC pins */
    bool  has_estop;                    /* Whether estop_pin is valid */
} nexus_safety_config_t;

/* ===================================================================
 * Reflex Binding
 * Matches: node_role_config.json#/definitions/reflex_binding
 * =================================================================== */

typedef struct {
    char name[NEXUS_REFLEX_NAME_MAX]; /* Reflex name */
    int  bytecode_offset;              /* Offset in bytecode memory */
    bool auto_start;                   /* Whether to auto-start on boot */
} nexus_reflex_binding_t;

/* ===================================================================
 * Baud Config
 * Matches: node_role_config.json#/definitions/baud_config
 * =================================================================== */

typedef struct {
    int  initial_baud;     /* Initial serial baud rate */
    bool negotiate;        /* Whether to negotiate baud rate */
} nexus_baud_config_t;

/* ===================================================================
 * Node Role Config (top-level)
 * Matches: node_role_config.json (root object)
 * =================================================================== */

typedef struct {
    char                        node_id[NEXUS_NODE_ID_MAX]; /* Unique node identifier */
    char                        role[NEXUS_ROLE_NAME_MAX];   /* Assigned role name */
    char                        firmware_version[NEXUS_VERSION_MAX]; /* Semver string */

    int                         num_sensors;
    nexus_sensor_pin_config_t   sensors[NEXUS_MAX_CHANNELS];

    int                         num_actuators;
    nexus_actuator_pin_config_t actuators[NEXUS_MAX_CHANNELS];

    nexus_safety_config_t       safety;          /* Safety configuration */

    int                         num_reflex_bindings;
    nexus_reflex_binding_t      reflex_bindings[NEXUS_REFLEX_BIND_MAX];

    nexus_baud_config_t         serial;          /* Serial config */
} nexus_node_role_config_t;

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_NODE_ROLE_CONFIG_H */
