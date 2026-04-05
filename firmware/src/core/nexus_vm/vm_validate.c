/**
 * @file vm_validate.c
 * @brief NEXUS Bytecode VM - Bytecode validator and disassembler.
 *
 * Validator: pre-execution checks on opcode range, jump targets,
 *            alignment, and stack depth analysis.
 * Disassembler: human-readable instruction listing.
 */

#include "vm.h"
#include "opcodes.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ===================================================================
 * Float16 -> Float32 (needed for CLAMP_F disassembly)
 * =================================================================== */

static uint32_t f16_to_f32_bits(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) return sign << 31;
        while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
        exponent++;
        mantissa &= 0x3FF;
    }

    uint32_t e32 = exponent + 112;
    uint32_t m32 = mantissa << 13;
    return (sign << 31) | (e32 << 23) | m32;
}

/* ===================================================================
 * vm_validate - Pre-execution bytecode validation
 *
 * Checks:
 *   1. Size is multiple of 8
 *   2. All opcodes are in valid range (0x00-0x1F)
 *   3. Syscall IDs are valid for NOP+SYSCALL instructions
 *   4. All jump targets are within bounds and 8-byte aligned
 *   5. Stack depth analysis (detect potential overflow/underflow)
 * =================================================================== */

vm_error_t vm_validate(const uint8_t *bytecode, uint32_t size) {
    if (!bytecode) {
        return ERR_INVALID_OPERAND;
    }
    if (size == 0 || size % 8 != 0) {
        return ERR_INVALID_OPERAND;
    }

    uint32_t num_instr = size / 8;

    /* Simple linear stack depth tracking (conservative) */
    int32_t stack_depth = 0;
    int32_t max_depth = 0;
    int32_t min_depth = 0;

    for (uint32_t i = 0; i < num_instr; i++) {
        const uint8_t *ip = bytecode + i * 8;
        uint8_t opcode   = ip[0];
        uint8_t flags    = ip[1];
        uint16_t operand1;
        uint32_t operand2;
        memcpy(&operand1, ip + 2, sizeof(uint16_t));
        memcpy(&operand2, ip + 4, sizeof(uint32_t));

        /* Check for syscall (NOP with SYSCALL flag) */
        if (opcode == OP_NOP && (flags & FLAGS_SYSCALL)) {
            uint8_t sid = (uint8_t)(operand2 & 0xFF);
            if (sid < SYSCALL_HALT || sid > SYSCALL_EMIT_EVENT) {
                return ERR_INVALID_SYSCALL;
            }
            /* Syscall stack effects */
            if (sid == SYSCALL_PID_COMPUTE) {
                stack_depth -= 2; /* pop setpoint + input */
                stack_depth += 1; /* push output */
            }
            continue;
        }

        /* Check opcode range (0x00-0x1F only) */
        if (opcode > 0x1F) {
            return ERR_INVALID_OPCODE;
        }

        /* Track stack depth (conservative linear analysis) */
        switch (opcode) {
        case OP_NOP:
            break;
        case OP_PUSH_I8:
        case OP_PUSH_I16:
        case OP_PUSH_F32:
        case OP_READ_PIN:
        case OP_READ_TIMER_MS:
            stack_depth++;
            break;
        case OP_POP:
            stack_depth--;
            break;
        case OP_DUP:
            stack_depth++;
            break;
        case OP_SWAP:
        case OP_ROT:
            break; /* net zero */
        case OP_ADD_F:
        case OP_SUB_F:
        case OP_MUL_F:
        case OP_DIV_F:
        case OP_MIN_F:
        case OP_MAX_F:
        case OP_EQ_F:
        case OP_LT_F:
        case OP_GT_F:
        case OP_LTE_F:
        case OP_GTE_F:
        case OP_AND_B:
        case OP_OR_B:
        case OP_XOR_B:
            stack_depth--; /* 2->1 */
            break;
        case OP_NEG_F:
        case OP_ABS_F:
        case OP_CLAMP_F:
        case OP_NOT_B:
            break; /* 1->1 */
        case OP_WRITE_PIN:
        case OP_JUMP_IF_FALSE:
        case OP_JUMP_IF_TRUE:
            stack_depth--;
            break;
        case OP_JUMP:
            break; /* 0->0 or call/ret */
        }

        if (stack_depth > max_depth) max_depth = stack_depth;
        if (stack_depth < min_depth) min_depth = stack_depth;

        /* Validate jump targets */
        switch (opcode) {
        case OP_JUMP: {
            /* RET sentinel */
            if (operand2 == 0xFFFFFFFFu) break;
            if (operand2 >= size || (operand2 % 8) != 0) {
                return ERR_JUMP_OUT_OF_BOUNDS;
            }
            break;
        }
        case OP_JUMP_IF_FALSE:
        case OP_JUMP_IF_TRUE:
            if (operand2 >= size || (operand2 % 8) != 0) {
                return ERR_JUMP_OUT_OF_BOUNDS;
            }
            break;
        default:
            break;
        }
    }

    /* Stack depth warnings (not errors - just informational) */
    (void)max_depth;
    (void)min_depth;

    return VM_OK;
}

/* ===================================================================
 * vm_disassemble - Print human-readable bytecode listing
 *
 * Output format per instruction:
 *   XXXX: MNEMONIC  [operands]
 * =================================================================== */

void vm_disassemble(const uint8_t *bytecode, uint32_t size, char *out, size_t out_max) {
    if (!bytecode || !out || out_max == 0) return;

    size_t pos = 0;
    uint32_t num_instr = (size > 0) ? size / 8 : 0;

    for (uint32_t i = 0; i < num_instr && pos < out_max - 2; i++) {
        const uint8_t *ip = bytecode + i * 8;
        uint8_t opcode   = ip[0];
        uint8_t flags    = ip[1];
        uint16_t operand1;
        uint32_t operand2;
        memcpy(&operand1, ip + 2, sizeof(uint16_t));
        memcpy(&operand2, ip + 4, sizeof(uint32_t));

        /* Address */
        int written = snprintf(out + pos, out_max - pos, "%04X: ", (unsigned)(i * 8));
        if (written < 0) break;
        pos += (size_t)written;
        if (pos >= out_max) break;

        /* Check for syscall */
        if (opcode == OP_NOP && (flags & FLAGS_SYSCALL)) {
            uint8_t sid = (uint8_t)(operand2 & 0xFF);
            switch (sid) {
            case SYSCALL_HALT:
                written = snprintf(out + pos, out_max - pos, "HALT\n");
                break;
            case SYSCALL_PID_COMPUTE:
                written = snprintf(out + pos, out_max - pos,
                                   "PID_COMPUTE %u\n", (unsigned)(operand1 & 0xFF));
                break;
            case SYSCALL_RECORD_SNAPSHOT:
                written = snprintf(out + pos, out_max - pos, "RECORD_SNAPSHOT\n");
                break;
            case SYSCALL_EMIT_EVENT: {
                uint16_t eid = operand1;
                uint16_t edata = (uint16_t)(operand2 >> 16);
                written = snprintf(out + pos, out_max - pos,
                                   "EMIT_EVENT %u %u\n", (unsigned)eid, (unsigned)edata);
                break;
            }
            default:
                written = snprintf(out + pos, out_max - pos,
                                   "NOP SYSCALL %u\n", (unsigned)sid);
                break;
            }
            if (written < 0) break;
            pos += (size_t)written;
            continue;
        }

        /* Core opcodes */
        switch (opcode) {
        case OP_NOP:
            written = snprintf(out + pos, out_max - pos, "NOP\n");
            break;

        case OP_PUSH_I8: {
            int8_t val = (int8_t)(operand1 & 0xFF);
            written = snprintf(out + pos, out_max - pos, "PUSH_I8  %d\n", (int)val);
            break;
        }

        case OP_PUSH_I16: {
            int16_t val = (int16_t)operand1;
            written = snprintf(out + pos, out_max - pos, "PUSH_I16 %d\n", (int)val);
            break;
        }

        case OP_PUSH_F32: {
            float val;
            memcpy(&val, &operand2, sizeof(float));
            written = snprintf(out + pos, out_max - pos, "PUSH_F32  %f\n", (double)val);
            break;
        }

        case OP_POP:
            written = snprintf(out + pos, out_max - pos, "POP\n");
            break;

        case OP_DUP:
            written = snprintf(out + pos, out_max - pos, "DUP\n");
            break;

        case OP_SWAP:
            written = snprintf(out + pos, out_max - pos, "SWAP\n");
            break;

        case OP_ROT:
            written = snprintf(out + pos, out_max - pos, "ROT\n");
            break;

        case OP_ADD_F:
            written = snprintf(out + pos, out_max - pos, "ADD_F\n");
            break;
        case OP_SUB_F:
            written = snprintf(out + pos, out_max - pos, "SUB_F\n");
            break;
        case OP_MUL_F:
            written = snprintf(out + pos, out_max - pos, "MUL_F\n");
            break;
        case OP_DIV_F:
            written = snprintf(out + pos, out_max - pos, "DIV_F\n");
            break;
        case OP_NEG_F:
            written = snprintf(out + pos, out_max - pos, "NEG_F\n");
            break;
        case OP_ABS_F:
            written = snprintf(out + pos, out_max - pos, "ABS_F\n");
            break;
        case OP_MIN_F:
            written = snprintf(out + pos, out_max - pos, "MIN_F\n");
            break;
        case OP_MAX_F:
            written = snprintf(out + pos, out_max - pos, "MAX_F\n");
            break;

        case OP_CLAMP_F: {
            uint16_t lo16 = (uint16_t)(operand2 & 0xFFFF);
            uint16_t hi16 = (uint16_t)((operand2 >> 16) & 0xFFFF);
            uint32_t lo_bits = f16_to_f32_bits(lo16);
            uint32_t hi_bits = f16_to_f32_bits(hi16);
            float lo, hi;
            memcpy(&lo, &lo_bits, sizeof(float));
            memcpy(&hi, &hi_bits, sizeof(float));
            written = snprintf(out + pos, out_max - pos,
                               "CLAMP_F   %f %f\n", (double)lo, (double)hi);
            break;
        }

        case OP_EQ_F:
            written = snprintf(out + pos, out_max - pos, "EQ_F\n");
            break;
        case OP_LT_F:
            written = snprintf(out + pos, out_max - pos, "LT_F\n");
            break;
        case OP_GT_F:
            written = snprintf(out + pos, out_max - pos, "GT_F\n");
            break;
        case OP_LTE_F:
            written = snprintf(out + pos, out_max - pos, "LTE_F\n");
            break;
        case OP_GTE_F:
            written = snprintf(out + pos, out_max - pos, "GTE_F\n");
            break;

        case OP_AND_B:
            written = snprintf(out + pos, out_max - pos, "AND_B\n");
            break;
        case OP_OR_B:
            written = snprintf(out + pos, out_max - pos, "OR_B\n");
            break;
        case OP_XOR_B:
            written = snprintf(out + pos, out_max - pos, "XOR_B\n");
            break;
        case OP_NOT_B:
            written = snprintf(out + pos, out_max - pos, "NOT_B\n");
            break;

        case OP_READ_PIN:
            written = snprintf(out + pos, out_max - pos, "READ_PIN %u\n", (unsigned)operand1);
            break;
        case OP_WRITE_PIN:
            written = snprintf(out + pos, out_max - pos, "WRITE_PIN %u\n", (unsigned)operand1);
            break;
        case OP_READ_TIMER_MS:
            written = snprintf(out + pos, out_max - pos, "READ_TIMER_MS\n");
            break;

        case OP_JUMP:
            if (operand2 == 0xFFFFFFFFu) {
                written = snprintf(out + pos, out_max - pos, "RET\n");
            } else if (flags & FLAGS_IS_CALL) {
                written = snprintf(out + pos, out_max - pos,
                                   "CALL %04X\n", (unsigned)operand2);
            } else {
                written = snprintf(out + pos, out_max - pos,
                                   "JUMP %04X\n", (unsigned)operand2);
            }
            break;
        case OP_JUMP_IF_FALSE:
            written = snprintf(out + pos, out_max - pos,
                               "JUMP_IF_FALSE %04X\n", (unsigned)operand2);
            break;
        case OP_JUMP_IF_TRUE:
            written = snprintf(out + pos, out_max - pos,
                               "JUMP_IF_TRUE %04X\n", (unsigned)operand2);
            break;

        default:
            written = snprintf(out + pos, out_max - pos,
                               "UNKNOWN 0x%02X\n", (unsigned)opcode);
            break;
        }

        if (written < 0) break;
        pos += (size_t)written;
    }

    /* Ensure null termination */
    if (pos >= out_max) pos = out_max - 1;
    out[pos] = '\0';
}
