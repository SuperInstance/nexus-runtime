/**
 * @file opcodes.h
 * @brief NEXUS Bytecode VM — Complete opcode definitions.
 *
 * Core opcodes (0x00-0x1F): 32 opcodes for the Reflex VM on ESP32-S3.
 * A2A opcodes (0x20-0x56): 29 opcodes for agent-to-agent communication.
 * A2A opcodes are NOP on existing ESP32 firmware (backward compatible).
 */

#ifndef NEXUS_OPCODES_H
#define NEXUS_OPCODES_H

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Core Opcodes (0x00-0x1F) - 32 opcodes
 * =================================================================== */

/* Stack operations (0x00-0x07) */
#define OP_NOP             0x00
#define OP_PUSH_I8         0x01
#define OP_PUSH_I16        0x02
#define OP_PUSH_F32        0x03
#define OP_POP             0x04
#define OP_DUP             0x05
#define OP_SWAP            0x06
#define OP_ROT             0x07

/* Arithmetic (0x08-0x10) */
#define OP_ADD_F           0x08
#define OP_SUB_F           0x09
#define OP_MUL_F           0x0A
#define OP_DIV_F           0x0B
#define OP_NEG_F           0x0C
#define OP_ABS_F           0x0D
#define OP_MIN_F           0x0E
#define OP_MAX_F           0x0F
#define OP_CLAMP_F         0x10

/* Comparison (0x11-0x15) */
#define OP_EQ_F            0x11
#define OP_LT_F            0x12
#define OP_GT_F            0x13
#define OP_LTE_F           0x14
#define OP_GTE_F           0x15

/* Logic (0x16-0x19) */
#define OP_AND_B           0x16
#define OP_OR_B            0x17
#define OP_XOR_B           0x18
#define OP_NOT_B           0x19

/* I/O (0x1A-0x1C) */
#define OP_READ_PIN        0x1A
#define OP_WRITE_PIN       0x1B
#define OP_READ_TIMER_MS   0x1C

/* Control flow (0x1D-0x1F) */
#define OP_JUMP            0x1D
#define OP_JUMP_IF_FALSE   0x1E
#define OP_JUMP_IF_TRUE    0x1F

/* ===================================================================
 * A2A Opcodes (0x20-0x56) - 29 opcodes
 * All NOP on ESP32 firmware (backward compatible).
 * =================================================================== */

/* Intent opcodes (0x20-0x26) */
#define OP_DECLARE_INTENT   0x20
#define OP_ASSERT_GOAL      0x21
#define OP_VERIFY_OUTCOME   0x22
#define OP_EXPLAIN_FAILURE  0x23

/* Agent Communication (0x30-0x34) */
#define OP_TELL             0x30
#define OP_ASK              0x31
#define OP_DELEGATE         0x32
#define OP_REPORT_STATUS    0x33
#define OP_REQUEST_OVERRIDE 0x34

/* Capability Negotiation (0x40-0x44) */
#define OP_REQUIRE_CAPABILITY    0x40
#define OP_DECLARE_SENSOR_NEED   0x41
#define OP_DECLARE_ACTUATOR_USE  0x42

/* Safety Augmentation (0x50-0x56) */
#define OP_TRUST_CHECK           0x50
#define OP_AUTONOMY_LEVEL_ASSERT 0x51
#define OP_SAFE_BOUNDARY         0x52
#define OP_RATE_LIMIT            0x53

/* ===================================================================
 * Opcode category helpers
 * =================================================================== */

#define OP_IS_CORE(op)    ((op) <= 0x1F)
#define OP_IS_A2A(op)     ((op) >= 0x20 && (op) <= 0x56)
#define OP_IS_STACK(op)   ((op) >= 0x00 && (op) <= 0x07)
#define OP_IS_ARITH(op)   ((op) >= 0x08 && (op) <= 0x10)
#define OP_IS_CMP(op)     ((op) >= 0x11 && (op) <= 0x15)
#define OP_IS_LOGIC(op)   ((op) >= 0x16 && (op) <= 0x19)
#define OP_IS_IO(op)      ((op) >= 0x1A && (op) <= 0x1C)
#define OP_IS_CONTROL(op) ((op) >= 0x1D && (op) <= 0x1F)

#define OPCODE_CORE_COUNT  32
#define OPCODE_A2A_COUNT   29
#define OPCODE_TOTAL_COUNT (OPCODE_CORE_COUNT + OPCODE_A2A_COUNT)

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_OPCODES_H */
