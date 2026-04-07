/**
 * @file vm.h
 *
 * Stack machine with 256-entry stack, float32 arithmetic,
 * 8-byte fixed instructions, and deterministic execution.
 * Zero heap allocation - all memory is statically sized.
 *
 * VM state total: ~5.4 KB (well within 512KB SRAM budget on ESP32-S3).
 */

#ifndef NEXUS_VM_H
#define NEXUS_VM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
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
#define VM_MAX_CYCLE_BUDGET   10000
#define VM_MAX_BYTECODE_SIZE  (100 * 1024)

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
} vm_error_t;

/* ===================================================================
 * PID Controller State (32 bytes per instance)
 * =================================================================== */

typedef struct {
    float Kp, Ki, Kd;
    float integral;
    float prev_error;
    float integral_limit;
    float output_min, output_max;
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
 * VM State - Complete interpreter state
 * =================================================================== */

typedef struct {
    uint32_t stack[VM_STACK_SIZE];
    uint16_t sp;
    uint32_t pc;
    uint32_t vars[VM_VAR_COUNT];
    uint32_t sensors[VM_SENSOR_COUNT];
    uint32_t actuators[VM_ACTUATOR_COUNT];
    uint32_t flags;
    uint32_t cycle_count;
    uint32_t cycle_budget;
    uint32_t tick_count_ms;
    float    tick_period_sec;

    struct {
        uint32_t return_addr;
        uint16_t frame_pointer;
    } call_stack[VM_CALL_STACK_SIZE];
    uint16_t csp;

    pid_state_t pid[VM_PID_COUNT];

    vm_snapshot_t snapshots[VM_SNAPSHOT_COUNT];
    uint8_t next_snapshot;

    vm_event_t events[VM_EVENT_RING_SIZE];
    uint16_t event_head;
    uint16_t event_tail;

    vm_error_t last_error;
    bool halted;
    const uint8_t *bytecode;
    uint32_t bytecode_size;
} vm_state_t;

_Static_assert(sizeof(vm_state_t) < 6000, "vm_state_t must be under 6KB");

/* ===================================================================
 * Core VM API
 * =================================================================== */

vm_error_t vm_init(vm_state_t *vm, const uint8_t *bytecode, uint32_t size);
vm_error_t vm_execute(vm_state_t *vm);
vm_error_t vm_tick(vm_state_t *vm);
vm_error_t vm_set_sensor(vm_state_t *vm, uint8_t idx, float value);
float vm_get_actuator(vm_state_t *vm, uint8_t idx);
vm_error_t vm_execute_instruction(vm_state_t *vm, const nexus_instruction_t *instr);
vm_error_t vm_validate(const uint8_t *bytecode, uint32_t size);
void vm_disassemble(const uint8_t *bytecode, uint32_t size, char *out, size_t out_max);

/* ===================================================================
 * Instruction Encoder Helpers
 * =================================================================== */

void encode_instruction(uint8_t *buf, uint8_t opcode, uint8_t flags,
                        uint16_t operand1, uint32_t operand2);
void encode_push_f32(uint8_t *buf, float value);
void encode_push_i8(uint8_t *buf, int8_t value);
void encode_push_i16(uint8_t *buf, int16_t value);
void encode_jump(uint8_t *buf, uint32_t target, bool is_call);
void encode_ret(uint8_t *buf);
void encode_syscall(uint8_t *buf, uint8_t syscall_id);
void encode_pid_compute(uint8_t *buf, uint8_t pid_index);
void encode_record_snapshot(uint8_t *buf);
void encode_emit_event(uint8_t *buf, uint16_t event_id, uint16_t event_data);
void encode_clamp_f(uint8_t *buf, float lo, float hi);
void encode_read_pin(uint8_t *buf, uint16_t pin);
void encode_write_pin(uint8_t *buf, uint16_t pin);
void encode_halt(uint8_t *buf);
void encode_nop(uint8_t *buf);
void encode_pop(uint8_t *buf);
void encode_dup(uint8_t *buf);
void encode_swap(uint8_t *buf);
void encode_rot(uint8_t *buf);
void encode_read_timer_ms(uint8_t *buf);
void encode_jump_if_true(uint8_t *buf, uint32_t target);
void encode_jump_if_false(uint8_t *buf, uint32_t target);
void encode_alu(uint8_t *buf, uint8_t opcode);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_VM_H */
