/**
 * @file unity.h
 * @brief Minimal Unity test framework stubs for host compilation.
 */

#ifndef UNITY_H
#define UNITY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef void (*UnityTestFunction)(void);

static int _unity_tests_run    = 0;
static int _unity_tests_failed = 0;
static int _unity_tests_passed = 0;
static int _unity_current_failed = 0;

#define TEST_ASSERT_PASS() do { } while(0)

#define TEST_ASSERT_FAIL_MESSAGE(msg) do { \
    printf("  FAIL: %s\n", msg); \
    _unity_current_failed++; \
} while(0)

#define TEST_ASSERT_TRUE(condition) do { \
    if (!(condition)) { \
        printf("  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        _unity_current_failed++; \
    } \
} while(0)

#define TEST_ASSERT_FALSE(condition) do { \
    if ((condition)) { \
        printf("  FAIL: %s:%d: NOT(%s)\n", __FILE__, __LINE__, #condition); \
        _unity_current_failed++; \
    } \
} while(0)

#define TEST_ASSERT_EQUAL_INT(expected, actual) do { \
    if ((int)(expected) != (int)(actual)) { \
        printf("  FAIL: %s:%d (expected %d, got %d)\n", \
               __FILE__, __LINE__, (int)(expected), (int)(actual)); \
        _unity_current_failed++; \
    } \
} while(0)

#define TEST_ASSERT_EQUAL_UINT16(expected, actual) TEST_ASSERT_EQUAL_INT(expected, actual)
#define TEST_ASSERT_EQUAL_UINT32(expected, actual) TEST_ASSERT_EQUAL_INT(expected, actual)
#define TEST_ASSERT_EQUAL_UINT8(expected, actual)  TEST_ASSERT_EQUAL_INT(expected, actual)
#define TEST_ASSERT_EQUAL_INT8(expected, actual)   TEST_ASSERT_EQUAL_INT(expected, actual)

#define TEST_ASSERT_EQUAL_FLOAT(expected, actual) do { \
    float _e = (float)(expected); \
    float _a = (float)(actual); \
    if (_e != _a && fabs((double)(_e - _a)) > 1e-5) { \
        printf("  FAIL: %s:%d (expected %f, got %f)\n", \
               __FILE__, __LINE__, (double)_e, (double)_a); \
        _unity_current_failed++; \
    } \
} while(0)

#define TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual) do { \
    float _e = (float)(expected); \
    float _a = (float)(actual); \
    float _d = (float)(delta); \
    if (fabs((double)(_e - _a)) > (double)_d) { \
        printf("  FAIL: %s:%d (expected %f +/- %f, got %f)\n", \
               __FILE__, __LINE__, (double)_e, (double)_d, (double)_a); \
        _unity_current_failed++; \
    } \
} while(0)

#define TEST_ASSERT_NULL(pointer) TEST_ASSERT_TRUE((pointer) == NULL)
#define TEST_ASSERT_NOT_NULL(pointer) TEST_ASSERT_TRUE((pointer) != NULL)

#define TEST_ASSERT_EQUAL_STRING(expected, actual) do { \
    const char *_e = (const char *)(expected); \
    const char *_a = (const char *)(actual); \
    if (strcmp(_e, _a) != 0) { \
        printf("  FAIL: %s:%d (expected \"%s\", got \"%s\")\n", \
               __FILE__, __LINE__, _e, _a); \
        _unity_current_failed++; \
    } \
} while(0)

#define TEST_CASE(name) void test_##name(void)

#define RUN_TEST(name) do { \
    _unity_tests_run++; \
    _unity_current_failed = 0; \
    printf("  [%02d] %-55s", _unity_tests_run, #name); \
    test_##name(); \
    if (_unity_current_failed == 0) { \
        printf("PASS\n"); \
        _unity_tests_passed++; \
    } \
} while(0)

static inline int unity_begin(void) {
    _unity_tests_run = 0;
    _unity_tests_passed = 0;
    _unity_tests_failed = 0;
    _unity_current_failed = 0;
    printf("=== Running VM Tests ===\n");
    return 0;
}

static inline int unity_end(void) {
    return _unity_current_failed + _unity_tests_failed;
}

#define UNITY_BEGIN() unity_begin()
#define UNITY_END() unity_end()

#endif /* UNITY_H */
