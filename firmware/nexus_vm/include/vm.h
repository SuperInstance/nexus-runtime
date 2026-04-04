/**
 * @file vm.h
 * @brief NEXUS Bytecode VM - Core data structures and constants.
 *
 * Stack machine with 256-entry stack, float32 arithmetic,
 * 8-byte fixed instructions, and deterministic execution.
 * Zero heap allocation - all memory is statically sized.
 */

#ifndef NEXUS_VM_H
#define NEXUS_VM_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "instruction.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * VM Constants
 * =================================================================== */

#define VM_STACK_SIZE         256
#define VM_CALL_STACK_SIZE    16
#define VM_VAR_COUNT          256
#define VM_SENSOR_COUNT       64
#define VM_ACTUATOR_COUNT     64
#define VM_PID_COUNT          8
#define VM_SNAPSHOT_COUNT     16
#define VM_EVENT_RING_SIZE    32
#define VM_MAX_CYCLE_BUDGET   100000

/* ===================================================================
 * Error Codes
 * =================================================================== */

typedef enum {
    VM_OK = 0,
    ERR_STACK_UNDERFLOW,
    ERR_STACK_OVERFLOW,
    ERR_INVALID_OPCODE,
    ERR_INVALID_OPERAND,
    ERR_JUMP_OUT_OF_BOUNDS,
    ERR_CALL_STACK_OVERFLOW,
    ERR_CALL_STACK_UNDERFLOW,
    ERR_CYCLE_BUDGET_EXCEEDED,
    ERR_INVALID_SYSCALL,
    ERR_INVALID_PID,
    ERR_DIVISION_BY_ZERO,
    ERR_NAN_IN_ACTUATOR,
} vm_error_t;

/* ===================================================================
 * PID Controller State (32 bytes per instance)
 * =================================================================== */

typedef struct {
    float kp;
    float ki;
    float kd;
    float integral;
    float prev_error;
    float integral_limit;
    float output_min;
    float output_max;
} pid_state_t;

/* ===================================================================
 * Snapshot (128 bytes per entry)
 * =================================================================== */

typedef struct {
    uint32_t tick_ms;
    uint32_t cycle_count;
    uint32_t current_state;
    uint32_t variables[15];
    uint32_t sensors[14];
} vm_snapshot_t;

/* ===================================================================
 * Event Ring Buffer Entry (8 bytes)
 * =================================================================== */

typedef struct {
    uint32_t tick_ms;
    uint16_t event_id;
    uint16_t event_data;
} vm_event_t;

/* ===================================================================
 * Call Stack Entry (6 bytes)
 * =================================================================== */

typedef struct {
    uint32_t return_addr;
    uint16_t frame_pointer;
} vm_call_frame_t;

/* ===================================================================
 * VM State - Complete interpreter state
 * =================================================================== */

typedef struct {
    /* Data stack */
    uint32_t stack[VM_STACK_SIZE];
    uint16_t sp;

    /* Program counter */
    uint32_t pc;

    /* Memory regions */
    uint32_t vars[VM_VAR_COUNT];
    uint32_t sensors[VM_SENSOR_COUNT];
    uint32_t actuators[VM_ACTUATOR_COUNT];

    /* Flags and counters */
    uint32_t flags;
    uint32_t cycle_count;
    uint32_t cycle_budget;
    uint32_t tick_count_ms;
    float    tick_period_sec;

    /* Call stack */
    vm_call_frame_t call_stack[VM_CALL_STACK_SIZE];
    uint16_t csp;

    /* PID controllers */
    pid_state_t pid[VM_PID_COUNT];

    /* Snapshots */
    vm_snapshot_t snapshots[VM_SNAPSHOT_COUNT];
    uint8_t next_snapshot;

    /* Event ring buffer */
    vm_event_t events[VM_EVENT_RING_SIZE];
    uint16_t event_head;
    uint16_t event_tail;

    /* State */
    vm_error_t last_error;
    bool halted;
    const uint8_t *bytecode;
    uint32_t bytecode_size;
} vm_state_t;

/* ===================================================================
 * VM API Functions
 * =================================================================== */

/**
 * @brief Initialize VM state to safe defaults.
 * @param vm Pointer to VM state structure.
 * @return VM_OK on success, error code on failure.
 */
vm_error_t vm_init(vm_state_t *vm);

/**
 * @brief Load bytecode into the VM.
 * @param vm Pointer to VM state.
 * @param bytecode Pointer to bytecode buffer.
 * @param size Size of bytecode in bytes (must be multiple of 8).
 * @return VM_OK on success, error code on failure.
 */
vm_error_t vm_load_bytecode(vm_state_t *vm, const uint8_t *bytecode, uint32_t size);

/**
 * @brief Execute one VM tick (fetch-decode-execute loop).
 * @param vm Pointer to VM state.
 * @return VM_OK on normal completion, error on fault.
 */
vm_error_t vm_execute_tick(vm_state_t *vm);

/**
 * @brief Validate bytecode before execution.
 * @param bytecode Pointer to bytecode buffer.
 * @param size Size of bytecode in bytes.
 * @return VM_OK if valid, error code if invalid.
 */
vm_error_t vm_validate_bytecode(const uint8_t *bytecode, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_VM_H */
