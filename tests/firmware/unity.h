/**
 * @file unity.h
 * @brief Minimal Unity test framework stubs for host compilation.
 */

#ifndef UNITY_H
#define UNITY_H

#include <stdio.h>
#include <stdlib.h>

typedef void (*UnityTestFunction)(void);

#define TEST_ASSERT_PASS() do { } while(0)
#define TEST_ASSERT_FAIL_MESSAGE(msg) do { printf("FAIL: %s\n", msg); } while(0)
#define TEST_ASSERT_TRUE(condition) do { if (!(condition)) { printf("FAIL: %s:%d\n", __FILE__, __LINE__); } } while(0)
#define TEST_ASSERT_FALSE(condition) do { if ((condition)) { printf("FAIL: %s:%d\n", __FILE__, __LINE__); } } while(0)
#define TEST_ASSERT_EQUAL_INT(expected, actual) do { if ((expected) != (actual)) { printf("FAIL: %s:%d (expected %d, got %d)\n", __FILE__, __LINE__, (int)(expected), (int)(actual)); } } while(0)
#define TEST_ASSERT_EQUAL_UINT16(expected, actual) TEST_ASSERT_EQUAL_INT(expected, actual)
#define TEST_ASSERT_EQUAL_UINT32(expected, actual) TEST_ASSERT_EQUAL_INT(expected, actual)
#define TEST_ASSERT_EQUAL_FLOAT(expected, actual) do { if ((expected) != (actual)) { printf("FAIL: %s:%d (expected %f, got %f)\n", __FILE__, __LINE__, (double)(expected), (double)(actual)); } } while(0)
#define TEST_ASSERT_NULL(pointer) TEST_ASSERT_TRUE((pointer) == NULL)
#define TEST_ASSERT_NOT_NULL(pointer) TEST_ASSERT_TRUE((pointer) != NULL)
#define TEST_ASSERT_EQUAL_INT8(expected, actual) TEST_ASSERT_EQUAL_INT(expected, actual)

#define TEST_CASE(name) void test_##name(void)
#define RUN_TEST(name) test_##name()

#define UNITY_BEGIN() 0
#define UNITY_END() 0

#endif /* UNITY_H */
