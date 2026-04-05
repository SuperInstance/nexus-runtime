/**
 * @file reflex_definition.h
 * @brief NEXUS Reflex Definition — C struct layouts matching reflex_definition.json schema.
 *
 * Structs for reflex instructions, PID configs, triggers, and reflex definitions.
 * Field names and types match the JSON schema definitions exactly.
 */

#ifndef NEXUS_REFLEX_DEFINITION_H
#define NEXUS_REFLEX_DEFINITION_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Constants
 * =================================================================== */

#define NEXUS_REFLEX_NAME_MAX    64
#define NEXUS_REFLEX_INTENT_MAX  128
#define NEXUS_MAX_SENSORS        16
#define NEXUS_MAX_ACTUATORS      16
#define NEXUS_MAX_BODY_SIZE      1000

/* Opcode names (string enum for JSON interop) */
typedef enum {
    NEXUS_OP_NOP              = 0,
    NEXUS_OP_PUSH_I8          = 1,
    NEXUS_OP_PUSH_I16         = 2,
    NEXUS_OP_PUSH_F32         = 3,
    NEXUS_OP_POP              = 4,
    NEXUS_OP_DUP              = 5,
    NEXUS_OP_SWAP             = 6,
    NEXUS_OP_ROT              = 7,
    NEXUS_OP_ADD_F            = 8,
    NEXUS_OP_SUB_F            = 9,
    NEXUS_OP_MUL_F            = 10,
    NEXUS_OP_DIV_F            = 11,
    NEXUS_OP_NEG_F            = 12,
    NEXUS_OP_ABS_F            = 13,
    NEXUS_OP_MIN_F            = 14,
    NEXUS_OP_MAX_F            = 15,
    NEXUS_OP_CLAMP_F          = 16,
    NEXUS_OP_EQ_F             = 17,
    NEXUS_OP_LT_F             = 18,
    NEXUS_OP_GT_F             = 19,
    NEXUS_OP_LTE_F            = 20,
    NEXUS_OP_GTE_F            = 21,
    NEXUS_OP_AND_B            = 22,
    NEXUS_OP_OR_B             = 23,
    NEXUS_OP_XOR_B            = 24,
    NEXUS_OP_NOT_B            = 25,
    NEXUS_OP_READ_PIN         = 26,
    NEXUS_OP_WRITE_PIN        = 27,
    NEXUS_OP_READ_TIMER_MS    = 28,
    NEXUS_OP_JUMP             = 29,
    NEXUS_OP_JUMP_IF_FALSE    = 30,
    NEXUS_OP_JUMP_IF_TRUE     = 31,
    /* A2A Intent */
    NEXUS_OP_DECLARE_INTENT   = 32,
    NEXUS_OP_ASSERT_GOAL      = 33,
    NEXUS_OP_VERIFY_OUTCOME   = 34,
    NEXUS_OP_EXPLAIN_FAILURE  = 35,
    NEXUS_OP_SET_PRIORITY     = 36,
    NEXUS_OP_REQUEST_RESOURCE = 37,
    NEXUS_OP_RELEASE_RESOURCE = 38,
    /* A2A Communication */
    NEXUS_OP_TELL             = 39,
    NEXUS_OP_ASK              = 40,
    NEXUS_OP_DELEGATE         = 41,
    NEXUS_OP_REPORT_STATUS    = 42,
    NEXUS_OP_REQUEST_OVERRIDE = 43,
    /* A2A Capability */
    NEXUS_OP_REQUIRE_CAPABILITY    = 44,
    NEXUS_OP_DECLARE_SENSOR_NEED   = 45,
    NEXUS_OP_DECLARE_ACTUATOR_USE  = 46,
    NEXUS_OP_CHECK_AVAILABILITY    = 47,
    NEXUS_OP_RESERVE_RESOURCE      = 48,
    /* A2A Safety */
    NEXUS_OP_TRUST_CHECK           = 49,
    NEXUS_OP_AUTONOMY_LEVEL_ASSERT = 50,
    NEXUS_OP_SAFE_BOUNDARY         = 51,
    NEXUS_OP_RATE_LIMIT            = 52,
    NEXUS_OP_EMERGENCY_CLAIM       = 53,
    NEXUS_OP_RELEASE_CLAIM         = 54,
    NEXUS_OP_VERIFY_AUTHORITY      = 55,
    NEXUS_OP_COUNT                 = 56,
} nexus_opcode_t;

/* ===================================================================
 * Reflex Instruction
 * Matches: reflex_definition.json#/definitions/instruction
 * =================================================================== */

typedef struct {
    nexus_opcode_t op;       /* Opcode */
    int32_t  arg;            /* Immediate integer operand (null -> INT32_MIN) */
    float    value;          /* Float immediate (null -> NAN) */
    uint8_t  flags;          /* Instruction flags */
    int32_t  operand1;       /* First operand */
    int32_t  operand2;       /* Second operand */
    float    lo;             /* CLAMP_F lower bound (null -> NAN) */
    float    hi;             /* CLAMP_F upper bound (null -> NAN) */
    char     target[32];     /* Jump target label (empty if null) */
    char     label[32];      /* Instruction label (empty if null) */
    bool     has_arg;        /* Whether arg is set */
    bool     has_value;      /* Whether value is set */
    bool     has_target;     /* Whether target is set */
    bool     has_label;      /* Whether label is set */
} nexus_instruction_def_t;

/* ===================================================================
 * PID Configuration
 * Matches: reflex_definition.json#/definitions/pid_config
 * =================================================================== */

typedef struct {
    float kp;           /* Proportional gain */
    float ki;           /* Integral gain */
    float kd;           /* Derivative gain */
    float output_min;   /* Minimum output clamp (NAN if not set) */
    float output_max;   /* Maximum output clamp (NAN if not set) */
    bool  has_limits;   /* Whether output_min/max are set */
} nexus_pid_config_t;

/* ===================================================================
 * Trigger Condition
 * Matches: reflex_definition.json#/definitions/trigger_condition
 * =================================================================== */

typedef enum {
    NEXUS_TRIGGER_ALWAYS             = 0,
    NEXUS_TRIGGER_SENSOR_THRESHOLD   = 1,
    NEXUS_TRIGGER_TIMER_MS           = 2,
    NEXUS_TRIGGER_TRUST_GATE         = 3,
} nexus_trigger_type_t;

typedef enum {
    NEXUS_CMP_GT   = 0,
    NEXUS_CMP_LT   = 1,
    NEXUS_CMP_GTE  = 2,
    NEXUS_CMP_LTE  = 3,
    NEXUS_CMP_EQ   = 4,
} nexus_comparator_t;

typedef struct {
    nexus_trigger_type_t type;      /* Trigger type */
    int32_t  sensor_id;             /* Sensor channel (null -> -1) */
    float    threshold;             /* Threshold value (NAN if not set) */
    nexus_comparator_t comparator;  /* Comparison operator */
    int32_t  interval_ms;           /* Timer interval (0 if not set) */
    float    min_trust;             /* Min trust score (NAN if not set) */
    int      min_level;             /* Min autonomy level (0 if not set) */
    bool     has_sensor_id;         /* Whether sensor_id is set */
    bool     has_threshold;         /* Whether threshold is set */
    bool     has_comparator;        /* Whether comparator is set */
    bool     has_interval_ms;       /* Whether interval_ms is set */
    bool     has_min_trust;         /* Whether min_trust is set */
    bool     has_min_level;         /* Whether min_level is set */
} nexus_trigger_condition_t;

/* ===================================================================
 * Reflex Author
 * Matches: reflex_definition.json#/properties/author
 * =================================================================== */

typedef enum {
    NEXUS_AUTHOR_HUMAN = 0,
    NEXUS_AUTHOR_AGENT = 1,
} nexus_author_t;

/* ===================================================================
 * Reflex Definition (top-level)
 * Matches: reflex_definition.json (root object)
 * =================================================================== */

typedef struct {
    char                     name[NEXUS_REFLEX_NAME_MAX];     /* Unique reflex identifier */
    char                     intent[NEXUS_REFLEX_INTENT_MAX]; /* Human-readable intent */
    char                     sensors[NEXUS_MAX_SENSORS][32];  /* Sensor input names */
    int                      num_sensors;
    char                     actuators[NEXUS_MAX_ACTUATORS][32]; /* Actuator output names */
    int                      num_actuators;
    float                    trust_min;            /* Min trust to deploy [0.0, 1.0] */
    nexus_author_t           author;               /* Author type */
    nexus_pid_config_t       pid;                  /* PID config (has_limits=false if absent) */
    bool                     has_pid;              /* Whether PID config is present */
    nexus_trigger_condition_t trigger;             /* Trigger condition */
    bool                     has_trigger;          /* Whether trigger is present */
    nexus_instruction_def_t  body[NEXUS_MAX_BODY_SIZE]; /* Instruction body */
    int                      body_size;            /* Number of instructions */
} nexus_reflex_definition_t;

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_REFLEX_DEFINITION_H */
