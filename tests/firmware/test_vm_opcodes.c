/**
 * @file test_vm_opcodes.c
 * @brief NEXUS VM opcode test vectors (stub).
 */

#include "unity.h"
#include "vm.h"
#include "opcodes.h"

TEST_CASE(vm_opcodes_nop) {
    vm_state_t vm;
    vm_error_t err = vm_init(&vm);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_FALSE(vm.halted);
    TEST_ASSERT_TRUE(vm.bytecode == NULL);
    TEST_ASSERT_EQUAL_INT(0, vm.sp);
}

TEST_CASE(vm_load_bytecode) {
    vm_state_t vm;
    vm_init(&vm);

    uint8_t bytecode[8] = {0};
    vm_error_t err = vm_load_bytecode(&vm, bytecode, 8);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
    TEST_ASSERT_TRUE(vm.bytecode != NULL);
    TEST_ASSERT_EQUAL_INT(8, vm.bytecode_size);
}

TEST_CASE(vm_validate_valid) {
    uint8_t bytecode[16] = {0};  /* Two NOP instructions */
    vm_error_t err = vm_validate_bytecode(bytecode, 16);
    TEST_ASSERT_EQUAL_INT(VM_OK, err);
}

TEST_CASE(vm_validate_invalid_size) {
    uint8_t bytecode[7] = {0};  /* Not multiple of 8 */
    vm_error_t err = vm_validate_bytecode(bytecode, 7);
    TEST_ASSERT_EQUAL_INT(ERR_INVALID_OPERAND, err);
}
