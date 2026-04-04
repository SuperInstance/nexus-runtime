/**
 * @file vm_opcodes.c
 * @brief NEXUS Bytecode VM - Per-opcode implementations and instruction encoders.
 *
 * Implements all 32 core opcodes (0x00-0x1F) and syscalls.
 * Also provides instruction encoder helper functions for building
 * bytecode programmatically.
 */

#include "vm.h"
#include "opcodes.h"
#include <string.h>
#include <math.h>

/* ===================================================================
 * Float16 <-> Float32 conversion helpers
 *
 * CLAMP_F encodes two float bounds as float16 in operand2:
 *   lower 16 bits = lo (float16)
 *   upper 16 bits = hi (float16)
 * =================================================================== */

static uint32_t f16_to_f32_bits(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            /* Zero */
            return sign << 31;
        }
        /* Denormal -> normalize */
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= 0x3FF;
    }

    uint32_t e32 = exponent + 112; /* bias: 127 - 15 = 112 */
    uint32_t m32 = mantissa << 13;

    return (sign << 31) | (e32 << 23) | m32;
}

static uint16_t f32_to_f16_bits(uint32_t f) {
    uint32_t sign     = (f >> 31) & 1;
    uint32_t exponent = (f >> 23) & 0xFF;
    uint32_t mantissa = f & 0x7FFFFF;

    if (exponent == 0) {
        return (uint16_t)(sign << 15); /* zero or denormal -> zero */
    }

    if (exponent == 255) {
        /* Inf or NaN */
        return (uint16_t)((sign << 15) | 0x7C00 | (mantissa >> 13));
    }

    int32_t e16 = (int32_t)exponent - 112;

    if (e16 <= 0) {
        return (uint16_t)(sign << 15); /* underflow -> zero */
    }
    if (e16 >= 31) {
        return (uint16_t)((sign << 15) | 0x7C00); /* overflow -> Inf */
    }

    uint16_t m16 = (uint16_t)(mantissa >> 13);
    return (uint16_t)((sign << 15) | ((uint16_t)e16 << 10) | m16);
}

/* ===================================================================
 * Sign extension helpers
 * =================================================================== */

static inline uint32_t sign_extend_8(uint8_t val) {
    return (uint32_t)(int32_t)(int8_t)val;
}

static inline uint32_t sign_extend_16(uint16_t val) {
    return (uint32_t)(int32_t)(int16_t)val;
}

/* ===================================================================
 * vm_execute_instruction - Execute a single instruction
 *
 * Dispatches based on opcode. For JUMP instructions, sets PC directly.
 * The caller (vm_core.c loop) advances PC by 8 only if PC was unchanged.
 * =================================================================== */

vm_error_t vm_execute_instruction(vm_state_t *vm, const nexus_instruction_t *instr) {
    if (!vm || !instr) {
        return ERR_INVALID_OPERAND;
    }

    uint8_t  opcode   = instr->opcode;
    uint8_t  flags    = instr->flags;
    uint16_t operand1 = instr->operand1;
    uint32_t operand2 = instr->operand2;

    /* -----------------------------------------------------------------
     * NOP with SYSCALL flag (flags bit 7 = 0x80)
     * ----------------------------------------------------------------- */
    if (opcode == OP_NOP && (flags & FLAGS_SYSCALL)) {
        uint8_t syscall_id = (uint8_t)(operand2 & 0xFF);

        switch (syscall_id) {
        case SYSCALL_HALT:
            vm->halted = true;
            return VM_OK;

        case SYSCALL_PID_COMPUTE: {
            if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
            float input, setpoint;
            memcpy(&input, &vm->stack[--vm->sp], sizeof(float));
            memcpy(&setpoint, &vm->stack[--vm->sp], sizeof(float));

            uint8_t pid_idx = (uint8_t)(operand1 & 0xFF);
            if (pid_idx >= VM_PID_COUNT) return ERR_INVALID_PID;

            pid_state_t *p = &vm->pid[pid_idx];
            float error     = setpoint - input;
            float derivative = error - p->prev_error;
            p->prev_error = error;

            /* Compute output using integral BEFORE this tick's accumulation */
            float output = p->Kp * error + p->Ki * p->integral + p->Kd * derivative;

            /* Accumulate integral AFTER output calculation (anti-windup) */
            p->integral += error;
            if (p->integral_limit != 0.0f) {
                if (p->integral > p->integral_limit)
                    p->integral = p->integral_limit;
                if (p->integral < -p->integral_limit)
                    p->integral = -p->integral_limit;
            }

            /* Clamp output */
            if (output < p->output_min) output = p->output_min;
            if (output > p->output_max) output = p->output_max;

            memcpy(&vm->stack[vm->sp++], &output, sizeof(float));
            return VM_OK;
        }

        case SYSCALL_RECORD_SNAPSHOT: {
            uint8_t slot = vm->next_snapshot;
            vm_snapshot_t *snap = &vm->snapshots[slot];
            snap->tick_ms      = vm->tick_count_ms;
            snap->cycle_count  = vm->cycle_count;
            snap->current_state = vm->pc;
            memcpy(snap->variables, vm->vars,
                   (15 < VM_VAR_COUNT ? 15 : VM_VAR_COUNT) * sizeof(uint32_t));
            memcpy(snap->sensors, vm->sensors,
                   (14 < VM_SENSOR_COUNT ? 14 : VM_SENSOR_COUNT) * sizeof(uint32_t));
            vm->next_snapshot = (slot + 1) % VM_SNAPSHOT_COUNT;
            return VM_OK;
        }

        case SYSCALL_EMIT_EVENT: {
            vm_event_t ev;
            ev.tick_ms    = vm->tick_count_ms;
            ev.event_id   = operand1;
            ev.event_data = (uint16_t)(operand2 >> 16);
            vm->events[vm->event_head] = ev;
            vm->event_head = (vm->event_head + 1) % VM_EVENT_RING_SIZE;
            return VM_OK;
        }

        default:
            return ERR_INVALID_SYSCALL;
        }
    }

    /* -----------------------------------------------------------------
     * Core opcodes (0x00-0x1F)
     * ----------------------------------------------------------------- */
    switch (opcode) {

    /* === Stack Operations (0x00-0x07) === */

    case OP_NOP:
        /* Plain NOP: no effect, PC advances by 8 in caller */
        break;

    case OP_PUSH_I8:
        if (vm->sp >= VM_STACK_SIZE) return ERR_STACK_OVERFLOW;
        vm->stack[vm->sp++] = sign_extend_8(operand1 & 0xFF);
        break;

    case OP_PUSH_I16:
        if (vm->sp >= VM_STACK_SIZE) return ERR_STACK_OVERFLOW;
        vm->stack[vm->sp++] = sign_extend_16(operand1);
        break;

    case OP_PUSH_F32:
        if (vm->sp >= VM_STACK_SIZE) return ERR_STACK_OVERFLOW;
        vm->stack[vm->sp++] = operand2;
        break;

    case OP_POP:
        if (vm->sp == 0) return ERR_STACK_UNDERFLOW;
        vm->sp--;
        break;

    case OP_DUP:
        if (vm->sp == 0) return ERR_STACK_UNDERFLOW;
        if (vm->sp >= VM_STACK_SIZE) return ERR_STACK_OVERFLOW;
        vm->stack[vm->sp] = vm->stack[vm->sp - 1];
        vm->sp++;
        break;

    case OP_SWAP: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        uint32_t tmp = vm->stack[vm->sp - 1];
        vm->stack[vm->sp - 1] = vm->stack[vm->sp - 2];
        vm->stack[vm->sp - 2] = tmp;
        break;
    }

    case OP_ROT: {
        if (vm->sp < 3) return ERR_STACK_UNDERFLOW;
        /* [..., C, B, A] -> [..., B, A, C] */
        uint32_t a = vm->stack[vm->sp - 1];
        uint32_t b = vm->stack[vm->sp - 2];
        uint32_t c = vm->stack[vm->sp - 3];
        vm->stack[vm->sp - 3] = b;
        vm->stack[vm->sp - 2] = a;
        vm->stack[vm->sp - 1] = c;
        break;
    }

    /* === Arithmetic (0x08-0x10) === */

    case OP_ADD_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        float r = a + b;
        memcpy(&vm->stack[vm->sp - 1], &r, sizeof(float));
        break;
    }

    case OP_SUB_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        float r = a - b;
        memcpy(&vm->stack[vm->sp - 1], &r, sizeof(float));
        break;
    }

    case OP_MUL_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        float r = a * b;
        memcpy(&vm->stack[vm->sp - 1], &r, sizeof(float));
        break;
    }

    case OP_DIV_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        float r = (b == 0.0f) ? 0.0f : (a / b);
        memcpy(&vm->stack[vm->sp - 1], &r, sizeof(float));
        break;
    }

    case OP_NEG_F:
        if (vm->sp < 1) return ERR_STACK_UNDERFLOW;
        vm->stack[vm->sp - 1] ^= 0x80000000u;
        break;

    case OP_ABS_F:
        if (vm->sp < 1) return ERR_STACK_UNDERFLOW;
        vm->stack[vm->sp - 1] &= 0x7FFFFFFFu;
        break;

    case OP_MIN_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        float r;
        if (isnan(a))      r = b;
        else if (isnan(b)) r = a;
        else               r = (a < b) ? a : b;
        memcpy(&vm->stack[vm->sp - 1], &r, sizeof(float));
        break;
    }

    case OP_MAX_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        float r;
        if (isnan(a))      r = b;
        else if (isnan(b)) r = a;
        else               r = (a > b) ? a : b;
        memcpy(&vm->stack[vm->sp - 1], &r, sizeof(float));
        break;
    }

    case OP_CLAMP_F: {
        if (vm->sp < 1) return ERR_STACK_UNDERFLOW;
        uint16_t lo16 = (uint16_t)(operand2 & 0xFFFF);
        uint16_t hi16 = (uint16_t)((operand2 >> 16) & 0xFFFF);
        uint32_t lo_bits = f16_to_f32_bits(lo16);
        uint32_t hi_bits = f16_to_f32_bits(hi16);
        float lo, hi, val;
        memcpy(&lo,  &lo_bits, sizeof(float));
        memcpy(&hi,  &hi_bits, sizeof(float));
        memcpy(&val, &vm->stack[vm->sp - 1], sizeof(float));
        if (val < lo) val = lo;
        if (val > hi) val = hi;
        memcpy(&vm->stack[vm->sp - 1], &val, sizeof(float));
        break;
    }

    /* === Comparison (0x11-0x15) === */

    case OP_EQ_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        vm->stack[vm->sp - 1] = (a == b) ? 1u : 0u;
        break;
    }

    case OP_LT_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        vm->stack[vm->sp - 1] = (a < b) ? 1u : 0u;
        break;
    }

    case OP_GT_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        vm->stack[vm->sp - 1] = (a > b) ? 1u : 0u;
        break;
    }

    case OP_LTE_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        vm->stack[vm->sp - 1] = (a <= b) ? 1u : 0u;
        break;
    }

    case OP_GTE_F: {
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        float b, a;
        memcpy(&b, &vm->stack[--vm->sp], sizeof(float));
        memcpy(&a, &vm->stack[vm->sp - 1], sizeof(float));
        vm->stack[vm->sp - 1] = (a >= b) ? 1u : 0u;
        break;
    }

    /* === Logic (0x16-0x19) === */

    case OP_AND_B:
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        vm->stack[vm->sp - 2] &= vm->stack[vm->sp - 1];
        vm->sp--;
        break;

    case OP_OR_B:
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        vm->stack[vm->sp - 2] |= vm->stack[vm->sp - 1];
        vm->sp--;
        break;

    case OP_XOR_B:
        if (vm->sp < 2) return ERR_STACK_UNDERFLOW;
        vm->stack[vm->sp - 2] ^= vm->stack[vm->sp - 1];
        vm->sp--;
        break;

    case OP_NOT_B:
        if (vm->sp < 1) return ERR_STACK_UNDERFLOW;
        vm->stack[vm->sp - 1] = ~vm->stack[vm->sp - 1];
        break;

    /* === I/O (0x1A-0x1C) === */

    case OP_READ_PIN: {
        if (vm->sp >= VM_STACK_SIZE) return ERR_STACK_OVERFLOW;
        if (operand1 < VM_SENSOR_COUNT) {
            vm->stack[vm->sp++] = vm->sensors[operand1];
        } else if (operand1 < VM_SENSOR_COUNT + VM_VAR_COUNT) {
            vm->stack[vm->sp++] = vm->vars[operand1 - VM_SENSOR_COUNT];
        } else {
            return ERR_INVALID_OPERAND;
        }
        break;
    }

    case OP_WRITE_PIN: {
        if (vm->sp == 0) return ERR_STACK_UNDERFLOW;
        uint32_t val = vm->stack[--vm->sp];
        if (operand1 < VM_ACTUATOR_COUNT) {
            vm->actuators[operand1] = val;
        } else if (operand1 < VM_ACTUATOR_COUNT + VM_VAR_COUNT) {
            vm->vars[operand1 - VM_ACTUATOR_COUNT] = val;
        } else {
            return ERR_INVALID_OPERAND;
        }
        break;
    }

    case OP_READ_TIMER_MS:
        if (vm->sp >= VM_STACK_SIZE) return ERR_STACK_OVERFLOW;
        vm->stack[vm->sp++] = vm->tick_count_ms;
        break;

    /* === Control Flow (0x1D-0x1F) === */

    case OP_JUMP: {
        /* RET: operand2 == 0xFFFFFFFF */
        if (operand2 == 0xFFFFFFFFu) {
            if (vm->csp == 0) return ERR_CALL_STACK_UNDERFLOW;
            vm->csp--;
            vm->pc = vm->call_stack[vm->csp].return_addr;
            vm->sp = vm->call_stack[vm->csp].frame_pointer;
            return VM_OK; /* PC already set */
        }

        /* CALL: flags bit 3 */
        if (flags & FLAGS_IS_CALL) {
            if (vm->csp >= VM_CALL_STACK_SIZE) return ERR_CALL_STACK_OVERFLOW;
            vm->call_stack[vm->csp].return_addr = vm->pc + 8;
            vm->call_stack[vm->csp].frame_pointer = vm->sp;
            vm->csp++;
        }

        /* Validate jump target */
        if (operand2 >= vm->bytecode_size || (operand2 % 8) != 0) {
            return ERR_JUMP_OUT_OF_BOUNDS;
        }

        vm->pc = operand2;
        return VM_OK; /* PC already set */
    }

    case OP_JUMP_IF_FALSE: {
        if (vm->sp == 0) return ERR_STACK_UNDERFLOW;
        uint32_t val = vm->stack[--vm->sp];
        if (val == 0) {
            if (operand2 >= vm->bytecode_size || (operand2 % 8) != 0) {
                return ERR_JUMP_OUT_OF_BOUNDS;
            }
            vm->pc = operand2;
            return VM_OK;
        }
        /* Not taken: advance to next instruction */
        vm->pc += 8;
        return VM_OK;
    }

    case OP_JUMP_IF_TRUE: {
        if (vm->sp == 0) return ERR_STACK_UNDERFLOW;
        uint32_t val = vm->stack[--vm->sp];
        if (val != 0) {
            if (operand2 >= vm->bytecode_size || (operand2 % 8) != 0) {
                return ERR_JUMP_OUT_OF_BOUNDS;
            }
            vm->pc = operand2;
            return VM_OK;
        }
        /* Not taken: advance to next instruction */
        vm->pc += 8;
        return VM_OK;
    }

    default:
        return ERR_INVALID_OPCODE;
    }

    return VM_OK;
}

/* ===================================================================
 * Instruction Encoder Helpers
 * =================================================================== */

void encode_instruction(uint8_t *buf, uint8_t opcode, uint8_t flags,
                        uint16_t operand1, uint32_t operand2) {
    buf[0] = opcode;
    buf[1] = flags;
    memcpy(buf + 2, &operand1, sizeof(uint16_t));
    memcpy(buf + 4, &operand2, sizeof(uint32_t));
}

void encode_push_f32(uint8_t *buf, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(float));
    encode_instruction(buf, OP_PUSH_F32, FLAGS_IS_FLOAT, 0, bits);
}

void encode_push_i8(uint8_t *buf, int8_t value) {
    encode_instruction(buf, OP_PUSH_I8, FLAGS_HAS_IMMEDIATE,
                      (uint16_t)(uint8_t)value, 0);
}

void encode_push_i16(uint8_t *buf, int16_t value) {
    encode_instruction(buf, OP_PUSH_I16, FLAGS_HAS_IMMEDIATE,
                      (uint16_t)value, 0);
}

void encode_jump(uint8_t *buf, uint32_t target, bool is_call) {
    encode_instruction(buf, OP_JUMP, is_call ? FLAGS_IS_CALL : (uint8_t)0,
                      0, target);
}

void encode_ret(uint8_t *buf) {
    encode_instruction(buf, OP_JUMP, 0, 0, 0xFFFFFFFFu);
}

void encode_syscall(uint8_t *buf, uint8_t syscall_id) {
    encode_instruction(buf, OP_NOP, FLAGS_SYSCALL, 0, (uint32_t)syscall_id);
}

void encode_pid_compute(uint8_t *buf, uint8_t pid_index) {
    encode_instruction(buf, OP_NOP, FLAGS_SYSCALL,
                      (uint16_t)pid_index, (uint32_t)SYSCALL_PID_COMPUTE);
}

void encode_record_snapshot(uint8_t *buf) {
    encode_syscall(buf, SYSCALL_RECORD_SNAPSHOT);
}

void encode_emit_event(uint8_t *buf, uint16_t event_id, uint16_t event_data) {
    uint32_t op2 = ((uint32_t)event_data << 16) | (uint32_t)SYSCALL_EMIT_EVENT;
    encode_instruction(buf, OP_NOP, FLAGS_SYSCALL, event_id, op2);
}

void encode_clamp_f(uint8_t *buf, float lo, float hi) {
    uint32_t lo_bits, hi_bits;
    memcpy(&lo_bits, &lo, sizeof(float));
    memcpy(&hi_bits, &hi, sizeof(float));
    uint16_t lo16 = f32_to_f16_bits(lo_bits);
    uint16_t hi16 = f32_to_f16_bits(hi_bits);
    uint32_t packed = ((uint32_t)hi16 << 16) | (uint32_t)lo16;
    encode_instruction(buf, OP_CLAMP_F, FLAGS_EXTENDED_CLAMP, 0, packed);
}

void encode_read_pin(uint8_t *buf, uint16_t pin) {
    encode_instruction(buf, OP_READ_PIN, 0, pin, 0);
}

void encode_write_pin(uint8_t *buf, uint16_t pin) {
    encode_instruction(buf, OP_WRITE_PIN, 0, pin, 0);
}

void encode_halt(uint8_t *buf) {
    encode_syscall(buf, SYSCALL_HALT);
}

void encode_nop(uint8_t *buf) {
    encode_instruction(buf, OP_NOP, 0, 0, 0);
}

void encode_pop(uint8_t *buf) {
    encode_instruction(buf, OP_POP, 0, 0, 0);
}

void encode_dup(uint8_t *buf) {
    encode_instruction(buf, OP_DUP, 0, 0, 0);
}

void encode_swap(uint8_t *buf) {
    encode_instruction(buf, OP_SWAP, 0, 0, 0);
}

void encode_rot(uint8_t *buf) {
    encode_instruction(buf, OP_ROT, 0, 0, 0);
}

void encode_read_timer_ms(uint8_t *buf) {
    encode_instruction(buf, OP_READ_TIMER_MS, 0, 0, 0);
}

void encode_jump_if_true(uint8_t *buf, uint32_t target) {
    encode_instruction(buf, OP_JUMP_IF_TRUE, 0, 0, target);
}

void encode_jump_if_false(uint8_t *buf, uint32_t target) {
    encode_instruction(buf, OP_JUMP_IF_FALSE, 0, 0, target);
}

void encode_alu(uint8_t *buf, uint8_t opcode) {
    encode_instruction(buf, opcode, 0, 0, 0);
}
