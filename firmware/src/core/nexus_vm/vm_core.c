/**
 * @file vm_core.c
 * @brief NEXUS Bytecode VM - Fetch-decode-execute loop and core API.
 *
 * Implements vm_init, vm_execute, vm_tick, vm_set_sensor, vm_get_actuator.
 * The execute loop fetches 8-byte instructions, dispatches to the
 * per-opcode handler in vm_opcodes.c, and manages cycle budgets.
 */

#include "vm.h"
#include "opcodes.h"
#include <string.h>

/* ===================================================================
 * Helpers
 * =================================================================== */

/** Check if a float (as uint32 bits) is NaN or Infinity. */
static inline bool vm_is_nan_or_inf_u32(uint32_t bits) {
    return (bits & 0x7F800000u) == 0x7F800000u;
}

static inline bool vm_is_nan_or_inf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return vm_is_nan_or_inf_u32(bits);
}

/* ===================================================================
 * vm_init - Initialize VM state and optionally load bytecode
 * =================================================================== */

vm_error_t vm_init(vm_state_t *vm, const uint8_t *bytecode, uint32_t size) {
    if (!vm) {
        return ERR_INVALID_OPERAND;
    }
    if (bytecode != NULL && size > 0 && size % 8 != 0) {
        return ERR_INVALID_OPERAND;
    }

    memset(vm, 0, sizeof(vm_state_t));
    vm->cycle_budget = VM_MAX_CYCLE_BUDGET;
    vm->bytecode = bytecode;
    vm->bytecode_size = size;
    vm->halted = false;
    vm->last_error = VM_OK;
    return VM_OK;
}

/* ===================================================================
 * vm_execute - Main fetch-decode-execute loop
 *
 * Runs until halted, error, cycle budget exceeded, or PC falls off
 * the end of bytecode. Each instruction consumes exactly 1 cycle
 * budget unit (regardless of opcode timing).
 * =================================================================== */

vm_error_t vm_execute(vm_state_t *vm) {
    if (!vm || !vm->bytecode) {
        return ERR_INVALID_OPERAND;
    }

    while (!vm->halted && vm->cycle_count < vm->cycle_budget) {
        /* Bounds check */
        if (vm->pc + 8 > vm->bytecode_size) {
            vm->halted = true;
            break;
        }

        /* Fetch 8-byte instruction */
        nexus_instruction_t instr;
        memcpy(&instr, vm->bytecode + vm->pc, sizeof(instr));
        uint8_t opcode = instr.opcode;

        /* Execute instruction */
        vm_error_t err = vm_execute_instruction(vm, &instr);
        vm->cycle_count++;

        if (err != VM_OK) {
            vm->last_error = err;
            vm->halted = true;
            return err;
        }

        /* Branch instructions manage PC themselves (set to target or advance).
         * Non-branch instructions need PC advanced by 8. */
        if (opcode != OP_JUMP && opcode != OP_JUMP_IF_FALSE && opcode != OP_JUMP_IF_TRUE) {
            vm->pc += 8;
        }
    }

    if (!vm->halted) {
        vm->last_error = ERR_CYCLE_BUDGET_EXCEEDED;
        vm->halted = true;
        return ERR_CYCLE_BUDGET_EXCEEDED;
    }

    return vm->last_error;
}

/* ===================================================================
 * vm_tick - Execute one tick (init + execute + post-tick)
 *
 * 1. Reset execution state (PC, SP, halted, cycles)
 * 2. Run vm_execute()
 * 3. Post-tick: NaN/Infinity guard on all actuator registers
 * =================================================================== */

vm_error_t vm_tick(vm_state_t *vm) {
    if (!vm) {
        return ERR_INVALID_OPERAND;
    }

    /* Reset execution state */
    vm->pc = 0;
    vm->sp = 0;
    vm->csp = 0;
    vm->halted = false;
    vm->cycle_count = 0;
    vm->last_error = VM_OK;

    /* Execute */
    vm_error_t err = vm_execute(vm);

    /* Post-tick: NaN/Infinity guard on all actuator registers */
    for (uint16_t i = 0; i < VM_ACTUATOR_COUNT; i++) {
        if (vm_is_nan_or_inf_u32(vm->actuators[i])) {
            vm->actuators[i] = 0;
        }
    }

    return err;
}

/* ===================================================================
 * vm_set_sensor - Set a sensor register value
 * =================================================================== */

vm_error_t vm_set_sensor(vm_state_t *vm, uint8_t idx, float value) {
    if (!vm) {
        return ERR_INVALID_OPERAND;
    }
    if (idx >= VM_SENSOR_COUNT) {
        return ERR_INVALID_OPERAND;
    }
    memcpy(&vm->sensors[idx], &value, sizeof(float));
    return VM_OK;
}

/* ===================================================================
 * vm_get_actuator - Read an actuator register value
 * =================================================================== */

float vm_get_actuator(vm_state_t *vm, uint8_t idx) {
    if (!vm || idx >= VM_ACTUATOR_COUNT) {
        return 0.0f;
    }
    float val;
    memcpy(&val, &vm->actuators[idx], sizeof(float));
    return val;
}
