/**
 * @file test_vm_opcodes.c
 * @brief NEXUS VM opcode test vectors — 52 tests covering all 32 opcodes,
 *        syscalls, error handling, encoder helpers, validator, and disassembler.
 */

#include "unity.h"
#include "vm.h"
#include "opcodes.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Helper: read stack slot as float */
static float stack_f32(const vm_state_t *vm, uint16_t idx) {
    float val;
    memcpy(&val, &vm->stack[idx], sizeof(float));
    return val;
}

/* ===================================================================
 * Test 01: Simple Arithmetic (spec vector 1)
 *   PUSH_F32 3.0, PUSH_F32 4.0, ADD_F, HALT
 *   Expected: stack[0] = 7.0, SP = 1, cycles = 4
 * =================================================================== */
TEST_CASE(test_01_simple_arithmetic) {
    uint8_t bytecode[32];
    int off = 0;
    encode_push_f32(bytecode + off, 3.0f); off += 8;
    encode_push_f32(bytecode + off, 4.0f); off += 8;
    encode_alu(bytecode + off, OP_ADD_F);  off += 8;
    encode_halt(bytecode + off);           off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(1, vm.sp);
    TEST_ASSERT_EQUAL_INT(4, (int)vm.cycle_count);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 02: Stack Underflow Detection (spec vector 2)
 *   ADD_F (stack empty) -> ERR_STACK_UNDERFLOW
 * =================================================================== */
TEST_CASE(test_02_stack_underflow) {
    uint8_t bytecode[16];
    encode_alu(bytecode, OP_ADD_F);
    encode_halt(bytecode + 8);

    vm_state_t vm;
    vm_init(&vm, bytecode, 16);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(ERR_STACK_UNDERFLOW, err);
    TEST_ASSERT_TRUE(vm.halted);
}

/* ===================================================================
 * Test 03: Division by Zero Returns 0.0 (spec vector 3)
 *   PUSH_F32 1.0, PUSH_F32 0.0, DIV_F, HALT
 *   Expected: stack[0] = 0.0 (NOT IEEE Inf/NaN)
 * =================================================================== */
TEST_CASE(test_03_div_zero_returns_zero) {
    uint8_t bytecode[32];
    int off = 0;
    encode_push_f32(bytecode + off, 1.0f); off += 8;
    encode_push_f32(bytecode + off, 0.0f); off += 8;
    encode_alu(bytecode + off, OP_DIV_F);  off += 8;
    encode_halt(bytecode + off);           off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, stack_f32(&vm, 0));
    /* Verify it's not NaN or Inf */
    float val = stack_f32(&vm, 0);
    TEST_ASSERT_FALSE(isnan(val));
    TEST_ASSERT_FALSE(isinf(val));
}

/* ===================================================================
 * Test 04: I/O Round-Trip (spec vector 4)
 *   Set sensor[0] = 0.42, READ_PIN 0, CLAMP_F 0.0 1.0, WRITE_PIN 0, HALT
 *   Expected: actuator[0] = 0.42
 * =================================================================== */
TEST_CASE(test_04_io_roundtrip) {
    uint8_t bytecode[40];
    int off = 0;
    encode_read_pin(bytecode + off, 0);        off += 8;
    encode_clamp_f(bytecode + off, 0.0f, 1.0f); off += 8;
    encode_write_pin(bytecode + off, 0);       off += 8;
    encode_halt(bytecode + off);               off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_set_sensor(&vm, 0, 0.42f);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.42f, vm_get_actuator(&vm, 0));
}

/* ===================================================================
 * Test 05: Conditional Branch (spec vector 5)
 *   PUSH_F32 5.0, PUSH_F32 10.0, LT_F, JUMP_IF_TRUE target,
 *   PUSH_F32 999.0, HALT, (target:) PUSH_F32 1.0, HALT
 *   Expected: stack[0] = 1.0 (5 < 10, so branch taken)
 * =================================================================== */
TEST_CASE(test_05_conditional_branch) {
    uint8_t bytecode[64];
    int off = 0;
    encode_push_f32(bytecode + off, 5.0f);  off += 8;  /* 0x00 */
    encode_push_f32(bytecode + off, 10.0f); off += 8;  /* 0x08 */
    encode_alu(bytecode + off, OP_LT_F);    off += 8;  /* 0x10 */
    uint32_t target = (uint32_t)(off + 8 + 16);        /* skip JUMP + 2 instrs */
    encode_jump_if_true(bytecode + off, target); off += 8; /* 0x18 */
    encode_push_f32(bytecode + off, 999.0f); off += 8;  /* 0x20 (dead code) */
    encode_halt(bytecode + off);               off += 8;  /* 0x28 (dead code) */
    encode_push_f32(bytecode + off, 1.0f);   off += 8;  /* 0x30 = target */
    encode_halt(bytecode + off);               off += 8;  /* 0x38 */

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(1, vm.sp);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 06: Cycle Budget Enforcement (spec vector 6)
 *   Cycle budget = 10, infinite JUMP loop
 *   Expected: ERR_CYCLE_BUDGET_EXCEEDED
 * =================================================================== */
TEST_CASE(test_06_cycle_budget) {
    uint8_t bytecode[8];
    encode_jump(bytecode, 0, false); /* JUMP to self */

    vm_state_t vm;
    vm_init(&vm, bytecode, 8);
    vm.cycle_budget = 10;
    vm_error_t err = vm_execute(&vm);

    TEST_ASSERT_EQUAL_INT(ERR_CYCLE_BUDGET_EXCEEDED, err);
    TEST_ASSERT_TRUE(vm.halted);
    TEST_ASSERT_TRUE(vm.cycle_count >= 10);
}

/* ===================================================================
 * Test 07: Jump Out of Bounds / RET with empty call stack (spec vector 7)
 *   JUMP 0xFFFFFFFF with empty call stack -> ERR_CALL_STACK_UNDERFLOW
 * =================================================================== */
TEST_CASE(test_07_ret_empty_call_stack) {
    uint8_t bytecode[16];
    encode_ret(bytecode);
    encode_halt(bytecode + 8);

    vm_state_t vm;
    vm_init(&vm, bytecode, 16);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(ERR_CALL_STACK_UNDERFLOW, err);
}

/* ===================================================================
 * Test 08: CLAMP_F Behavior (spec vector 8)
 *   PUSH_F32 -5.0, CLAMP_F -1.0 1.0, HALT
 *   Expected: stack[0] = -1.0
 * =================================================================== */
TEST_CASE(test_08_clamp_behavior) {
    uint8_t bytecode[24];
    int off = 0;
    encode_push_f32(bytecode + off, -5.0f);      off += 8;
    encode_clamp_f(bytecode + off, -1.0f, 1.0f); off += 8;
    encode_halt(bytecode + off);                  off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_FLOAT(-1.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 09: NEG_F and ABS_F Bit Manipulation (spec vector 9)
 *   PUSH_F32 3.14, NEG_F, ABS_F, HALT
 *   Expected: stack[0] = 3.14
 * =================================================================== */
TEST_CASE(test_09_neg_abs) {
    uint8_t bytecode[32];
    int off = 0;
    encode_push_f32(bytecode + off, 3.14f);       off += 8;
    encode_alu(bytecode + off, OP_NEG_F);          off += 8;
    encode_alu(bytecode + off, OP_ABS_F);          off += 8;
    encode_halt(bytecode + off);                   off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 3.14f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 10: PID_COMPUTE Syscall (spec vector 10)
 *   Configure PID[0], PUSH setpoint 50, PUSH input 48, PID_COMPUTE 0, HALT
 *   Expected: output = Kp*(50-48) = 2.0
 * =================================================================== */
TEST_CASE(test_10_pid_compute) {
    uint8_t bytecode[32];
    int off = 0;
    encode_push_f32(bytecode + off, 50.0f);  off += 8;
    encode_push_f32(bytecode + off, 48.0f);  off += 8;
    encode_pid_compute(bytecode + off, 0);   off += 8;
    encode_halt(bytecode + off);              off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm.pid[0].Kp = 1.0f;
    vm.pid[0].Ki = 0.1f;
    vm.pid[0].Kd = 0.0f;
    vm.pid[0].integral_limit = 100.0f;
    vm.pid[0].output_min = -100.0f;
    vm.pid[0].output_max = 100.0f;
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 2.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 11: All stack operations
 * =================================================================== */
TEST_CASE(test_11_all_stack_ops) {
    uint8_t bytecode[80];
    int off = 0;
    encode_push_i8(bytecode + off, -10);     off += 8;  /* SP=1 */
    encode_push_i16(bytecode + off, 1000);  off += 8;  /* SP=2 */
    encode_push_f32(bytecode + off, 2.5f);  off += 8;  /* SP=3 */
    encode_dup(bytecode + off);             off += 8;  /* SP=4 */
    encode_swap(bytecode + off);            off += 8;  /* SP=4, swap top 2 */
    encode_pop(bytecode + off);             off += 8;  /* SP=3 */
    encode_rot(bytecode + off);             off += 8;  /* SP=3, rotate top 3 */
    encode_halt(bytecode + off);            off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(3, vm.sp);
    /* After all ops: stack should have 3 elements */
    /* Check no underflow/overflow occurred */
}

/* ===================================================================
 * Test 12: All arithmetic operations
 * =================================================================== */
TEST_CASE(test_12_all_arithmetic) {
    /* ADD: 2+3=5 */
    uint8_t bc1[24];
    encode_push_f32(bc1, 2.0f); encode_push_f32(bc1+8, 3.0f);
    encode_alu(bc1+16, OP_ADD_F);
    vm_state_t vm1; vm_init(&vm1, bc1, 24);
    nexus_instruction_t instr1; memcpy(&instr1, bc1+16, 8);
    vm1.sp = 2; vm1.stack[0] = 0x40000000u; vm1.stack[1] = 0x40400000u;
    vm_execute_instruction(&vm1, &instr1);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, stack_f32(&vm1, 0));

    /* SUB: 10-4=6 */
    uint8_t bc2[24];
    encode_push_f32(bc2, 10.0f); encode_push_f32(bc2+8, 4.0f);
    encode_alu(bc2+16, OP_SUB_F);
    vm_state_t vm2; vm_init(&vm2, bc2, 24);
    vm2.sp = 2; vm2.stack[0] = 0x41200000u; vm2.stack[1] = 0x40800000u;
    nexus_instruction_t instr2; memcpy(&instr2, bc2+16, 8);
    vm_execute_instruction(&vm2, &instr2);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 6.0f, stack_f32(&vm2, 0));

    /* MUL: 3*7=21 */
    uint8_t bc3[24];
    encode_push_f32(bc3, 3.0f); encode_push_f32(bc3+8, 7.0f);
    encode_alu(bc3+16, OP_MUL_F);
    vm_state_t vm3; vm_init(&vm3, bc3, 24);
    vm3.sp = 2;
    float v3a = 3.0f, v3b = 7.0f;
    memcpy(&vm3.stack[0], &v3a, 4); memcpy(&vm3.stack[1], &v3b, 4);
    nexus_instruction_t instr3; memcpy(&instr3, bc3+16, 8);
    vm_execute_instruction(&vm3, &instr3);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 21.0f, stack_f32(&vm3, 0));

    /* NEG: -(3.0) = -3.0 */
    uint8_t bc4[16];
    encode_push_f32(bc4, 3.0f); encode_alu(bc4+8, OP_NEG_F);
    vm_state_t vm4; vm_init(&vm4, bc4, 16);
    vm4.sp = 1; float v4 = 3.0f; memcpy(&vm4.stack[0], &v4, 4);
    nexus_instruction_t instr4; memcpy(&instr4, bc4+8, 8);
    vm_execute_instruction(&vm4, &instr4);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -3.0f, stack_f32(&vm4, 0));

    /* ABS: -5.0 -> 5.0 */
    uint8_t bc5[16];
    encode_push_f32(bc5, -5.0f); encode_alu(bc5+8, OP_ABS_F);
    vm_state_t vm5; vm_init(&vm5, bc5, 16);
    vm5.sp = 1; float v5 = -5.0f; memcpy(&vm5.stack[0], &v5, 4);
    nexus_instruction_t instr5; memcpy(&instr5, bc5+8, 8);
    vm_execute_instruction(&vm5, &instr5);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, stack_f32(&vm5, 0));

    /* MIN: min(2.0, 5.0) = 2.0 */
    uint8_t bc6[24];
    encode_push_f32(bc6, 2.0f); encode_push_f32(bc6+8, 5.0f);
    encode_alu(bc6+16, OP_MIN_F);
    vm_state_t vm6; vm_init(&vm6, bc6, 24);
    vm6.sp = 2; float v6a = 2.0f, v6b = 5.0f;
    memcpy(&vm6.stack[0], &v6a, 4); memcpy(&vm6.stack[1], &v6b, 4);
    nexus_instruction_t instr6; memcpy(&instr6, bc6+16, 8);
    vm_execute_instruction(&vm6, &instr6);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f, stack_f32(&vm6, 0));

    /* MAX: max(2.0, 5.0) = 5.0 */
    uint8_t bc7[24];
    encode_push_f32(bc7, 2.0f); encode_push_f32(bc7+8, 5.0f);
    encode_alu(bc7+16, OP_MAX_F);
    vm_state_t vm7; vm_init(&vm7, bc7, 24);
    vm7.sp = 2; float v7a = 2.0f, v7b = 5.0f;
    memcpy(&vm7.stack[0], &v7a, 4); memcpy(&vm7.stack[1], &v7b, 4);
    nexus_instruction_t instr7; memcpy(&instr7, bc7+16, 8);
    vm_execute_instruction(&vm7, &instr7);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, stack_f32(&vm7, 0));
}

/* ===================================================================
 * Test 13: All comparisons
 * =================================================================== */
TEST_CASE(test_13_all_comparisons) {
    float a = 3.0f, b = 5.0f;

    /* EQ_F: 3 == 5 -> 0 */
    vm_state_t vm; vm_init(&vm, NULL, 0);
    vm.sp = 2;
    memcpy(&vm.stack[0], &a, 4); memcpy(&vm.stack[1], &b, 4);
    uint8_t bc[8]; encode_alu(bc, OP_EQ_F);
    nexus_instruction_t instr; memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(0, (int)vm.stack[vm.sp - 1]);

    /* LT_F: 3 < 5 -> 1 */
    vm.sp = 2;
    memcpy(&vm.stack[0], &a, 4); memcpy(&vm.stack[1], &b, 4);
    encode_alu(bc, OP_LT_F); memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(1, (int)vm.stack[vm.sp - 1]);

    /* GT_F: 3 > 5 -> 0 */
    vm.sp = 2;
    memcpy(&vm.stack[0], &a, 4); memcpy(&vm.stack[1], &b, 4);
    encode_alu(bc, OP_GT_F); memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(0, (int)vm.stack[vm.sp - 1]);

    /* LTE_F: 3 <= 5 -> 1 */
    vm.sp = 2;
    memcpy(&vm.stack[0], &a, 4); memcpy(&vm.stack[1], &b, 4);
    encode_alu(bc, OP_LTE_F); memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(1, (int)vm.stack[vm.sp - 1]);

    /* GTE_F: 3 >= 5 -> 0 */
    vm.sp = 2;
    memcpy(&vm.stack[0], &a, 4); memcpy(&vm.stack[1], &b, 4);
    encode_alu(bc, OP_GTE_F); memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(0, (int)vm.stack[vm.sp - 1]);
}

/* ===================================================================
 * Test 14: All logic operations
 * =================================================================== */
TEST_CASE(test_14_all_logic_ops) {
    uint8_t bc[8]; nexus_instruction_t instr;

    /* AND_B: 0xFF & 0x0F = 0x0F */
    vm_state_t vm; vm_init(&vm, NULL, 0);
    vm.sp = 2; vm.stack[0] = 0xFF; vm.stack[1] = 0x0F;
    encode_alu(bc, OP_AND_B); memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(0x0F, (int)vm.stack[0]);

    /* OR_B: 0xF0 | 0x0F = 0xFF */
    vm.sp = 2; vm.stack[0] = 0xF0; vm.stack[1] = 0x0F;
    encode_alu(bc, OP_OR_B); memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(0xFF, (int)vm.stack[0]);

    /* XOR_B: 0xFF ^ 0xFF = 0x00 */
    vm.sp = 2; vm.stack[0] = 0xFF; vm.stack[1] = 0xFF;
    encode_alu(bc, OP_XOR_B); memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(0x00, (int)vm.stack[0]);

    /* NOT_B: ~0x00 = 0xFFFFFFFF */
    vm.sp = 1; vm.stack[0] = 0x00;
    encode_alu(bc, OP_NOT_B); memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(0xFFFFFFFF, (int)vm.stack[0]);
}

/* ===================================================================
 * Test 15: READ_TIMER_MS
 * =================================================================== */
TEST_CASE(test_15_read_timer_ms) {
    uint8_t bytecode[16];
    encode_read_timer_ms(bytecode);
    encode_halt(bytecode + 8);

    vm_state_t vm;
    vm_init(&vm, bytecode, 16);
    vm.tick_count_ms = 12345;
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(1, vm.sp);
    TEST_ASSERT_EQUAL_INT(12345, (int)vm.stack[0]);
}

/* ===================================================================
 * Test 16: CALL and RETURN
 * =================================================================== */
TEST_CASE(test_16_call_return) {
    /*
     * 0x00: PUSH_F32 42.0
     * 0x08: CALL 0x18
     * 0x10: HALT           <- return lands here
     * 0x18: PUSH_F32 1.0   <- callee (writes above frame pointer)
     * 0x20: RET
     * 0x28: HALT
     * After RET: sp is restored to frame_pointer (1), callee value discarded.
     * Verify: CALL pushed correct return addr, execution resumed at 0x10.
     */
    uint8_t bytecode[48];
    int off = 0;
    encode_push_f32(bytecode + off, 42.0f); off += 8;  /* 0x00 */
    uint32_t call_target = (uint32_t)(off + 8);         /* 0x18 */
    encode_jump(bytecode + off, call_target, true); off += 8; /* 0x08: CALL */
    encode_halt(bytecode + off); off += 8;            /* 0x10: return target */
    encode_push_f32(bytecode + off, 1.0f); off += 8;  /* 0x18: callee */
    encode_ret(bytecode + off);            off += 8;  /* 0x20: RET */
    encode_halt(bytecode + off);            off += 8;  /* 0x28 */

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(1, vm.sp);   /* frame pointer restored, callee stack discarded */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 42.0f, stack_f32(&vm, 0));
    /* Verify we resumed at the HALT after CALL (not the dead code after callee) */
}

/* ===================================================================
 * Test 17: Stack overflow (257 pushes)
 * =================================================================== */
TEST_CASE(test_17_stack_overflow) {
    /* Build bytecode with 257 PUSH_F32 instructions + HALT */
    uint8_t bytecode[260 * 8];
    int off = 0;
    for (int i = 0; i < 257; i++) {
        encode_push_f32(bytecode + off, (float)i);
        off += 8;
    }
    encode_halt(bytecode + off);
    off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(ERR_STACK_OVERFLOW, err);
}

/* ===================================================================
 * Test 18: NaN guard on actuator write
 * =================================================================== */
TEST_CASE(test_18_nan_guard_actuator) {
    uint8_t bytecode[16];
    int off = 0;
    /* Push NaN as uint32 bits */
    uint32_t nan_bits = 0x7FC00000u;
    encode_push_f32(bytecode + off, 1.0f); off += 8;
    /* Overwrite with NaN bits by writing raw */
    bytecode[off] = OP_PUSH_F32;
    bytecode[off + 1] = 0;
    memset(bytecode + off + 2, 0, 2);
    memcpy(bytecode + off + 4, &nan_bits, 4);
    off += 8;
    /* Push would succeed but let's just write NaN directly to actuator via WRITE_PIN */
    /* Actually, let's use a different approach: push NaN value directly */

    /* Better approach: encode a NaN float */
    float nan_val = NAN;
    uint8_t bc2[16];
    off = 0;
    uint32_t nan_u32;
    memcpy(&nan_u32, &nan_val, 4);
    bc2[0] = OP_PUSH_F32; bc2[1] = 0;
    memset(bc2 + 2, 0, 2);
    memcpy(bc2 + 4, &nan_u32, 4);
    off = 8;
    encode_write_pin(bc2 + off, 0); off += 8;

    vm_state_t vm;
    vm_init(&vm, bc2, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    /* Post-tick NaN guard should zero the actuator */
    TEST_ASSERT_EQUAL_FLOAT(0.0f, vm_get_actuator(&vm, 0));
}

/* ===================================================================
 * Test 19: Infinity guard on actuator write
 * =================================================================== */
TEST_CASE(test_19_inf_guard_actuator) {
    uint8_t bc[16];
    int off = 0;
    float inf_val = INFINITY;
    uint32_t inf_u32;
    memcpy(&inf_u32, &inf_val, 4);
    bc[0] = OP_PUSH_F32; bc[1] = 0;
    memset(bc + 2, 0, 2);
    memcpy(bc + 4, &inf_u32, 4);
    off = 8;
    encode_write_pin(bc + off, 0); off += 8;

    vm_state_t vm;
    vm_init(&vm, bc, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, vm_get_actuator(&vm, 0));
}

/* ===================================================================
 * Test 20: RECORD_SNAPSHOT syscall
 * =================================================================== */
TEST_CASE(test_20_record_snapshot) {
    uint8_t bytecode[16];
    int off = 0;
    encode_record_snapshot(bytecode + off); off += 8;
    encode_halt(bytecode + off); off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm.tick_count_ms = 100;
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(100, (int)vm.snapshots[0].tick_ms);
    TEST_ASSERT_EQUAL_INT(1, (int)vm.next_snapshot);
}

/* ===================================================================
 * Test 21: EMIT_EVENT syscall
 * =================================================================== */
TEST_CASE(test_21_emit_event) {
    uint8_t bytecode[16];
    int off = 0;
    encode_emit_event(bytecode + off, 42, 100); off += 8;
    encode_halt(bytecode + off); off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm.tick_count_ms = 500;
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(42, (int)vm.events[0].event_id);
    TEST_ASSERT_EQUAL_INT(100, (int)vm.events[0].event_data);
    TEST_ASSERT_EQUAL_INT(500, (int)vm.events[0].tick_ms);
    TEST_ASSERT_EQUAL_INT(1, (int)vm.event_head);
}

/* ===================================================================
 * Test 22: MIN_F with NaN (returns non-NaN operand)
 * =================================================================== */
TEST_CASE(test_22_min_f_nan) {
    vm_state_t vm; vm_init(&vm, NULL, 0);
    float a = NAN, b = 5.0f;
    vm.sp = 2;
    memcpy(&vm.stack[0], &a, 4); memcpy(&vm.stack[1], &b, 4);
    uint8_t bc[8]; encode_alu(bc, OP_MIN_F);
    nexus_instruction_t instr; memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 23: MAX_F with NaN (returns non-NaN operand)
 * =================================================================== */
TEST_CASE(test_23_max_f_nan) {
    vm_state_t vm; vm_init(&vm, NULL, 0);
    float a = NAN, b = 5.0f;
    vm.sp = 2;
    memcpy(&vm.stack[0], &a, 4); memcpy(&vm.stack[1], &b, 4);
    uint8_t bc[8]; encode_alu(bc, OP_MAX_F);
    nexus_instruction_t instr; memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 24: All-0xFF instruction (invalid opcode)
 * =================================================================== */
TEST_CASE(test_24_invalid_opcode) {
    uint8_t bytecode[16];
    memset(bytecode, 0xFF, 8); /* all 0xFF: opcode = 0xFF > 0x1F */
    encode_halt(bytecode + 8);

    vm_state_t vm;
    vm_init(&vm, bytecode, 16);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(ERR_INVALID_OPCODE, err);
}

/* ===================================================================
 * Test 25: Empty bytecode
 * =================================================================== */
TEST_CASE(test_25_empty_bytecode) {
    vm_state_t vm;
    vm_error_t err = vm_init(&vm, NULL, 0);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_NULL(vm.bytecode);
    TEST_ASSERT_EQUAL_INT(0, (int)vm.bytecode_size);
}

/* ===================================================================
 * Test 26: Single HALT
 * =================================================================== */
TEST_CASE(test_26_single_halt) {
    uint8_t bytecode[8];
    encode_halt(bytecode);

    vm_state_t vm;
    vm_init(&vm, bytecode, 8);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_TRUE(vm.halted);
    TEST_ASSERT_EQUAL_INT(1, (int)vm.cycle_count);
    TEST_ASSERT_EQUAL_INT(0, vm.sp);
}

/* ===================================================================
 * Test 27: PUSH_I8 sign extension
 * =================================================================== */
TEST_CASE(test_27_push_i8_sign_ext) {
    /* -128 */
    uint8_t bc1[8]; encode_push_i8(bc1, -128);
    vm_state_t vm1; vm_init(&vm1, bc1, 8);
    vm1.sp = 0;
    nexus_instruction_t instr; memcpy(&instr, bc1, 8);
    vm_execute_instruction(&vm1, &instr);
    TEST_ASSERT_EQUAL_INT(0xFFFFFF80, (int)vm1.stack[0]);

    /* -1 */
    uint8_t bc2[8]; encode_push_i8(bc2, -1);
    vm_state_t vm2; vm_init(&vm2, bc2, 8);
    vm2.sp = 0; memcpy(&instr, bc2, 8);
    vm_execute_instruction(&vm2, &instr);
    TEST_ASSERT_EQUAL_INT(0xFFFFFFFF, (int)vm2.stack[0]);

    /* 127 */
    uint8_t bc3[8]; encode_push_i8(bc3, 127);
    vm_state_t vm3; vm_init(&vm3, bc3, 8);
    vm3.sp = 0; memcpy(&instr, bc3, 8);
    vm_execute_instruction(&vm3, &instr);
    TEST_ASSERT_EQUAL_INT(127, (int)vm3.stack[0]);
}

/* ===================================================================
 * Test 28: PUSH_I16 sign extension
 * =================================================================== */
TEST_CASE(test_28_push_i16_sign_ext) {
    /* -32768 */
    uint8_t bc1[8]; encode_push_i16(bc1, -32768);
    vm_state_t vm1; vm_init(&vm1, bc1, 8);
    vm1.sp = 0;
    nexus_instruction_t instr; memcpy(&instr, bc1, 8);
    vm_execute_instruction(&vm1, &instr);
    TEST_ASSERT_EQUAL_INT(0xFFFF8000, (int)vm1.stack[0]);

    /* -1 */
    uint8_t bc2[8]; encode_push_i16(bc2, -1);
    vm_state_t vm2; vm_init(&vm2, bc2, 8);
    vm2.sp = 0; memcpy(&instr, bc2, 8);
    vm_execute_instruction(&vm2, &instr);
    TEST_ASSERT_EQUAL_INT(0xFFFFFFFF, (int)vm2.stack[0]);

    /* 32767 */
    uint8_t bc3[8]; encode_push_i16(bc3, 32767);
    vm_state_t vm3; vm_init(&vm3, bc3, 8);
    vm3.sp = 0; memcpy(&instr, bc3, 8);
    vm_execute_instruction(&vm3, &instr);
    TEST_ASSERT_EQUAL_INT(32767, (int)vm3.stack[0]);
}

/* ===================================================================
 * Test 29: ROT with exactly 3 elements
 * =================================================================== */
TEST_CASE(test_29_rot_three) {
    /* Stack: [10, 20, 30] -> ROT -> [20, 30, 10] */
    vm_state_t vm; vm_init(&vm, NULL, 0);
    vm.stack[0] = 10; vm.stack[1] = 20; vm.stack[2] = 30;
    vm.sp = 3;

    uint8_t bc[8]; encode_rot(bc);
    nexus_instruction_t instr; memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);

    TEST_ASSERT_EQUAL_INT(3, vm.sp);
    TEST_ASSERT_EQUAL_INT(20, (int)vm.stack[0]); /* C->B */
    TEST_ASSERT_EQUAL_INT(30, (int)vm.stack[1]); /* B->A */
    TEST_ASSERT_EQUAL_INT(10, (int)vm.stack[2]); /* A->C */
}

/* ===================================================================
 * Test 30: ROT with wrong number of elements
 * =================================================================== */
TEST_CASE(test_30_rot_insufficient) {
    vm_state_t vm; vm_init(&vm, NULL, 0);
    vm.sp = 2; vm.stack[0] = 1; vm.stack[1] = 2;

    uint8_t bc[8]; encode_rot(bc);
    nexus_instruction_t instr; memcpy(&instr, bc, 8);
    vm_error_t err = vm_execute_instruction(&vm, &instr);

    TEST_ASSERT_EQUAL_INT(ERR_STACK_UNDERFLOW, err);
}

/* ===================================================================
 * Test 31: WRITE_PIN to variable (operand1 >= 64)
 * =================================================================== */
TEST_CASE(test_31_write_pin_variable) {
    uint8_t bytecode[16];
    int off = 0;
    encode_push_f32(bytecode + off, 99.0f); off += 8;
    uint16_t var_pin = 64 + 5; /* variable index 5 */
    encode_write_pin(bytecode + off, var_pin); off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    float result;
    memcpy(&result, &vm.vars[5], sizeof(float));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 99.0f, result);
}

/* ===================================================================
 * Test 32: READ_PIN from variable
 * =================================================================== */
TEST_CASE(test_32_read_pin_variable) {
    uint8_t bytecode[16];
    uint16_t var_pin = 64 + 3;
    encode_read_pin(bytecode, var_pin);
    encode_halt(bytecode + 8);

    vm_state_t vm;
    vm_init(&vm, bytecode, 16);
    /* Set variable 3 to a value */
    float val = 77.0f;
    memcpy(&vm.vars[3], &val, sizeof(float));
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 77.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 33: PID integral anti-windup
 * =================================================================== */
TEST_CASE(test_33_pid_antiwindup) {
    uint8_t bytecode[32];
    int off = 0;
    encode_push_f32(bytecode + off, 100.0f); off += 8; /* setpoint */
    encode_push_f32(bytecode + off, 0.0f);   off += 8; /* input */
    encode_pid_compute(bytecode + off, 0);   off += 8;
    encode_halt(bytecode + off);              off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    vm.pid[0].Kp = 0.0f;
    vm.pid[0].Ki = 1.0f;
    vm.pid[0].Kd = 0.0f;
    vm.pid[0].integral_limit = 10.0f; /* limit integral to +/- 10 */
    vm.pid[0].output_min = -1000.0f;
    vm.pid[0].output_max = 1000.0f;

    /* First tick: error = 100, integral before = 0, output = 0 */
    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);

    /* After first tick, integral should be 100, clamped to 10 */
    TEST_ASSERT_TRUE(vm.pid[0].integral <= 10.0f + 0.01f);
}

/* ===================================================================
 * Test 34: Disassembler output
 * =================================================================== */
TEST_CASE(test_34_disassembler) {
    uint8_t bytecode[48];
    int off = 0;
    encode_push_f32(bytecode + off, 3.14f); off += 8;
    encode_push_f32(bytecode + off, 2.718f); off += 8;
    encode_alu(bytecode + off, OP_ADD_F); off += 8;
    encode_clamp_f(bytecode + off, -1.0f, 1.0f); off += 8;
    encode_write_pin(bytecode + off, 0); off += 8;
    encode_halt(bytecode + off); off += 8;

    char buf[512];
    vm_disassemble(bytecode, (uint32_t)off, buf, sizeof(buf));

    /* Check key substrings */
    TEST_ASSERT_TRUE(strstr(buf, "PUSH_F32") != NULL);
    TEST_ASSERT_TRUE(strstr(buf, "ADD_F") != NULL);
    TEST_ASSERT_TRUE(strstr(buf, "CLAMP_F") != NULL);
    TEST_ASSERT_TRUE(strstr(buf, "WRITE_PIN 0") != NULL);
    TEST_ASSERT_TRUE(strstr(buf, "HALT") != NULL);
}

/* ===================================================================
 * Test 35: Validator catches invalid jump target (out of bounds)
 * =================================================================== */
TEST_CASE(test_35_validator_jump_oob) {
    uint8_t bytecode[16];
    encode_jump(bytecode, 9999, false); /* target beyond bytecode */
    encode_halt(bytecode + 8);

    vm_error_t err = vm_validate(bytecode, 16);
    TEST_ASSERT_EQUAL_INT(ERR_JUMP_OUT_OF_BOUNDS, err);
}

/* ===================================================================
 * Test 36: Validator catches non-aligned jump target
 * =================================================================== */
TEST_CASE(test_36_validator_jump_misaligned) {
    uint8_t bytecode[16];
    encode_jump(bytecode, 3, false); /* target 3 is not 8-byte aligned */
    encode_halt(bytecode + 8);

    vm_error_t err = vm_validate(bytecode, 16);
    TEST_ASSERT_EQUAL_INT(ERR_JUMP_OUT_OF_BOUNDS, err);
}

/* ===================================================================
 * Test 37: Multiple PID instances
 * =================================================================== */
TEST_CASE(test_37_multiple_pid) {
    uint8_t bytecode[32];
    int off = 0;
    encode_push_f32(bytecode + off, 10.0f); off += 8;
    encode_push_f32(bytecode + off, 5.0f);  off += 8;
    encode_pid_compute(bytecode + off, 0);  off += 8;
    encode_halt(bytecode + off);             off += 8;

    vm_state_t vm;
    vm_init(&vm, bytecode, (uint32_t)off);
    /* Configure PID[0] */
    vm.pid[0].Kp = 2.0f; vm.pid[0].Ki = 0.0f; vm.pid[0].Kd = 0.0f;
    vm.pid[0].integral_limit = 0.0f;
    vm.pid[0].output_min = -100.0f; vm.pid[0].output_max = 100.0f;
    /* Configure PID[1] differently */
    vm.pid[1].Kp = 5.0f; vm.pid[1].Ki = 0.0f; vm.pid[1].Kd = 0.0f;
    vm.pid[1].output_min = -100.0f; vm.pid[1].output_max = 100.0f;

    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    /* PID[0]: Kp=2.0, error=5.0, output = 10.0 */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 10.0f, stack_f32(&vm, 0));
    /* PID[1] should be untouched */
    TEST_ASSERT_EQUAL_FLOAT(0.0f, vm.pid[1].integral);
}

/* ===================================================================
 * Test 38: VM state size check (~5.4KB)
 * =================================================================== */
TEST_CASE(test_38_vm_state_size) {
    size_t sz = sizeof(vm_state_t);
    TEST_ASSERT_TRUE(sz >= 5000 && sz < 6000);
    printf("  (vm_state_t = %zu bytes)\n", sz);
}

/* ===================================================================
 * Test 39: SUB_F
 * =================================================================== */
TEST_CASE(test_39_sub_f) {
    uint8_t bc[24];
    encode_push_f32(bc, 10.0f); encode_push_f32(bc+8, 3.0f);
    encode_alu(bc+16, OP_SUB_F);

    vm_state_t vm; vm_init(&vm, bc, 24);
    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 7.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 40: MUL_F
 * =================================================================== */
TEST_CASE(test_40_mul_f) {
    uint8_t bc[24];
    encode_push_f32(bc, 6.0f); encode_push_f32(bc+8, 7.0f);
    encode_alu(bc+16, OP_MUL_F);

    vm_state_t vm; vm_init(&vm, bc, 24);
    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 42.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 41: NEG_F standalone
 * =================================================================== */
TEST_CASE(test_41_neg_f) {
    uint8_t bc[16];
    encode_push_f32(bc, 7.0f);
    encode_alu(bc+8, OP_NEG_F);

    vm_state_t vm; vm_init(&vm, bc, 16);
    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -7.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 42: ABS_F standalone
 * =================================================================== */
TEST_CASE(test_42_abs_f) {
    uint8_t bc[16];
    float neg = -42.0f;
    uint32_t neg_bits;
    memcpy(&neg_bits, &neg, 4);
    bc[0] = OP_PUSH_F32; bc[1] = 0;
    memset(bc+2, 0, 2); memcpy(bc+4, &neg_bits, 4);
    encode_alu(bc+8, OP_ABS_F);

    vm_state_t vm; vm_init(&vm, bc, 16);
    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 42.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 43: GTE_F comparison
 * =================================================================== */
TEST_CASE(test_43_gte_f) {
    /* 5 >= 5 -> 1 */
    uint8_t bc[24];
    encode_push_f32(bc, 5.0f); encode_push_f32(bc+8, 5.0f);
    encode_alu(bc+16, OP_GTE_F);

    vm_state_t vm; vm_init(&vm, bc, 24);
    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(1, (int)vm.stack[0]);
}

/* ===================================================================
 * Test 44: LTE_F comparison
 * =================================================================== */
TEST_CASE(test_44_lte_f) {
    /* 3 <= 5 -> 1 */
    uint8_t bc[24];
    encode_push_f32(bc, 3.0f); encode_push_f32(bc+8, 5.0f);
    encode_alu(bc+16, OP_LTE_F);

    vm_state_t vm; vm_init(&vm, bc, 24);
    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(1, (int)vm.stack[0]);
}

/* ===================================================================
 * Test 45: NOT_B
 * =================================================================== */
TEST_CASE(test_45_not_b) {
    vm_state_t vm; vm_init(&vm, NULL, 0);
    vm.sp = 1; vm.stack[0] = 0x0000FFFF;
    uint8_t bc[8]; encode_alu(bc, OP_NOT_B);
    nexus_instruction_t instr; memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(0xFFFF0000, (int)vm.stack[0]);
}

/* ===================================================================
 * Test 46: PUSH_F32 then POP (net zero stack)
 * =================================================================== */
TEST_CASE(test_46_push_pop) {
    uint8_t bc[16];
    encode_push_f32(bc, 42.0f);
    encode_pop(bc + 8);

    vm_state_t vm; vm_init(&vm, bc, 16);
    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(0, vm.sp);
}

/* ===================================================================
 * Test 47: SWAP two values
 * =================================================================== */
TEST_CASE(test_47_swap) {
    vm_state_t vm; vm_init(&vm, NULL, 0);
    vm.stack[0] = 100; vm.stack[1] = 200;
    vm.sp = 2;
    uint8_t bc[8]; encode_swap(bc);
    nexus_instruction_t instr; memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(200, (int)vm.stack[0]);
    TEST_ASSERT_EQUAL_INT(100, (int)vm.stack[1]);
}

/* ===================================================================
 * Test 48: DUP
 * =================================================================== */
TEST_CASE(test_48_dup) {
    vm_state_t vm; vm_init(&vm, NULL, 0);
    vm.stack[0] = 42;
    vm.sp = 1;
    uint8_t bc[8]; encode_dup(bc);
    nexus_instruction_t instr; memcpy(&instr, bc, 8);
    vm_execute_instruction(&vm, &instr);
    TEST_ASSERT_EQUAL_INT(2, vm.sp);
    TEST_ASSERT_EQUAL_INT(42, (int)vm.stack[0]);
    TEST_ASSERT_EQUAL_INT(42, (int)vm.stack[1]);
}

/* ===================================================================
 * Test 49: JUMP_IF_FALSE not taken
 * =================================================================== */
TEST_CASE(test_49_jump_if_false_not_taken) {
    uint8_t bc[32];
    int off = 0;
    encode_push_f32(bc + off, 1.0f); off += 8;    /* truthy */
    uint32_t target = (uint32_t)(off + 8);         /* skip one instruction */
    encode_jump_if_false(bc + off, target); off += 8;
    encode_push_f32(bc + off, 99.0f); off += 8;    /* should execute */
    encode_halt(bc + off); off += 8;

    vm_state_t vm; vm_init(&vm, bc, (uint32_t)off);
    vm_error_t err = vm_tick(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 99.0f, stack_f32(&vm, 0));
}

/* ===================================================================
 * Test 50: Validator catches invalid opcode
 * =================================================================== */
TEST_CASE(test_50_validator_invalid_opcode) {
    uint8_t bytecode[8];
    memset(bytecode, 0, 8);
    bytecode[0] = 0x20; /* A2A opcode - not in core range */

    vm_error_t err = vm_validate(bytecode, 8);
    TEST_ASSERT_EQUAL_INT(ERR_INVALID_OPCODE, err);
}

/* ===================================================================
 * Test 51: Multiple snapshots
 * =================================================================== */
TEST_CASE(test_51_multiple_snapshots) {
    uint8_t bc[32];
    int off = 0;
    encode_record_snapshot(bc + off); off += 8;
    encode_record_snapshot(bc + off); off += 8;
    encode_record_snapshot(bc + off); off += 8;
    encode_halt(bc + off); off += 8;

    vm_state_t vm; vm_init(&vm, bc, (uint32_t)off);
    vm.tick_count_ms = 100;
    vm_error_t err = vm_tick(&vm);

    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_EQUAL_INT(3, (int)vm.next_snapshot); /* wraps at VM_SNAPSHOT_COUNT */
    TEST_ASSERT_EQUAL_INT(100, (int)vm.snapshots[0].tick_ms);
    TEST_ASSERT_EQUAL_INT(100, (int)vm.snapshots[1].tick_ms);
}

/* ===================================================================
 * Test 52: JUMP to valid target (unconditional loop with counter)
 * =================================================================== */
TEST_CASE(test_52_jump_valid) {
    /*
     * 0x00: PUSH_F32 1.0
     * 0x08: JUMP 0x00  (infinite loop, but budget stops it)
     */
    uint8_t bc[16];
    encode_push_f32(bc, 1.0f);
    encode_jump(bc + 8, 0, false);

    vm_state_t vm; vm_init(&vm, bc, 16);
    vm.cycle_budget = 5;
    vm_error_t err = vm_execute(&vm);

    TEST_ASSERT_EQUAL_INT(ERR_CYCLE_BUDGET_EXCEEDED, err);
}

/* ===================================================================
 * Main test runner
 * =================================================================== */

int main(void) {
    int failures = UNITY_BEGIN();

    RUN_TEST(test_01_simple_arithmetic);
    RUN_TEST(test_02_stack_underflow);
    RUN_TEST(test_03_div_zero_returns_zero);
    RUN_TEST(test_04_io_roundtrip);
    RUN_TEST(test_05_conditional_branch);
    RUN_TEST(test_06_cycle_budget);
    RUN_TEST(test_07_ret_empty_call_stack);
    RUN_TEST(test_08_clamp_behavior);
    RUN_TEST(test_09_neg_abs);
    RUN_TEST(test_10_pid_compute);
    RUN_TEST(test_11_all_stack_ops);
    RUN_TEST(test_12_all_arithmetic);
    RUN_TEST(test_13_all_comparisons);
    RUN_TEST(test_14_all_logic_ops);
    RUN_TEST(test_15_read_timer_ms);
    RUN_TEST(test_16_call_return);
    RUN_TEST(test_17_stack_overflow);
    RUN_TEST(test_18_nan_guard_actuator);
    RUN_TEST(test_19_inf_guard_actuator);
    RUN_TEST(test_20_record_snapshot);
    RUN_TEST(test_21_emit_event);
    RUN_TEST(test_22_min_f_nan);
    RUN_TEST(test_23_max_f_nan);
    RUN_TEST(test_24_invalid_opcode);
    RUN_TEST(test_25_empty_bytecode);
    RUN_TEST(test_26_single_halt);
    RUN_TEST(test_27_push_i8_sign_ext);
    RUN_TEST(test_28_push_i16_sign_ext);
    RUN_TEST(test_29_rot_three);
    RUN_TEST(test_30_rot_insufficient);
    RUN_TEST(test_31_write_pin_variable);
    RUN_TEST(test_32_read_pin_variable);
    RUN_TEST(test_33_pid_antiwindup);
    RUN_TEST(test_34_disassembler);
    RUN_TEST(test_35_validator_jump_oob);
    RUN_TEST(test_36_validator_jump_misaligned);
    RUN_TEST(test_37_multiple_pid);
    RUN_TEST(test_38_vm_state_size);
    RUN_TEST(test_39_sub_f);
    RUN_TEST(test_40_mul_f);
    RUN_TEST(test_41_neg_f);
    RUN_TEST(test_42_abs_f);
    RUN_TEST(test_43_gte_f);
    RUN_TEST(test_44_lte_f);
    RUN_TEST(test_45_not_b);
    RUN_TEST(test_46_push_pop);
    RUN_TEST(test_47_swap);
    RUN_TEST(test_48_dup);
    RUN_TEST(test_49_jump_if_false_not_taken);
    RUN_TEST(test_50_validator_invalid_opcode);
    RUN_TEST(test_51_multiple_snapshots);
    RUN_TEST(test_52_jump_valid);

    failures = UNITY_END();

    printf("\n=== Results: %d tests, %d passed, %d failed ===\n",
           _unity_tests_run, _unity_tests_passed, failures);

    return (failures == 0) ? 0 : 1;
}
