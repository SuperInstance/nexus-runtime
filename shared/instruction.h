/**
 * @file instruction.h
 * @brief NEXUS Bytecode VM - 8-byte packed instruction format.
 *
 * Every instruction is exactly 8 bytes, fixed-length, cache-line aligned.
 * All multi-byte fields are little-endian on ESP32-S3 (Xtensa LX7).
 */

#ifndef NEXUS_INSTRUCTION_H
#define NEXUS_INSTRUCTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 8-byte fixed instruction format (packed, no padding).
 *
 * Layout:
 *   Byte 0:    opcode     (uint8)  - 0x00 to 0x56
 *   Byte 1:    flags      (uint8)  - bit field
 *   Bytes 2-3: operand1   (uint16) - little-endian
 *   Bytes 4-7: operand2   (uint32) - little-endian
 */
typedef struct __attribute__((packed)) {
    uint8_t  opcode;    /* Byte 0: 0x00-0x56 */
    uint8_t  flags;     /* Byte 1: bit field */
    uint16_t operand1;  /* Bytes 2-3: uint16 (little-endian) */
    uint32_t operand2;  /* Bytes 4-7: uint32 (little-endian) */
} nexus_instruction_t;

/* Verify the struct is exactly 8 bytes */
#define NEXUS_INSTRUCTION_SIZE sizeof(nexus_instruction_t)

/* ===================================================================
 * Flags bit definitions
 * =================================================================== */

/** Bit 0: Instruction has an immediate operand embedded */
#define FLAGS_HAS_IMMEDIATE   (1 << 0)

/** Bit 1: Operand fields contain a float32 value */
#define FLAGS_IS_FLOAT        (1 << 1)

/** Bit 2: CLAMP_F uses extended (separate lo/hi) encoding */
#define FLAGS_EXTENDED_CLAMP  (1 << 2)

/** Bit 3: JUMP is a CALL (push return address) */
#define FLAGS_IS_CALL         (1 << 3)

/** Bit 4-6: Reserved for future use */
#define FLAGS_RESERVED_MASK   0x78

/** Bit 7: NOP with SYSCALL flag set - syscall dispatch */
#define FLAGS_SYSCALL         (1 << 7)

/* ===================================================================
 * CLAMP_F operand encoding
 *
 * Standard:   operand2 = float_as_uint32 (single bound from operand1)
 * Extended:   operand2 = (hi16 << 16) | lo16, each half is uint16
 *             mapped to float via a separate conversion table
 * =================================================================== */

/* ===================================================================
 * Syscall IDs (used when opcode=OP_NOP, flags=FLAGS_SYSCALL)
 * =================================================================== */

#define SYSCALL_HALT            0x01
#define SYSCALL_PID_COMPUTE     0x02
#define SYSCALL_RECORD_SNAPSHOT 0x03
#define SYSCALL_EMIT_EVENT      0x04

/* ===================================================================
 * Helper: build a HALT instruction
 * =================================================================== */

static inline nexus_instruction_t nexus_make_halt(void) {
    nexus_instruction_t instr;
    instr.opcode   = 0x00; /* NOP */
    instr.flags    = FLAGS_SYSCALL;
    instr.operand1 = 0x0000;
    instr.operand2 = SYSCALL_HALT;
    return instr;
}

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_INSTRUCTION_H */
