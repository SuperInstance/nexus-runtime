/**
 * @file nexus_main.c
 * @brief NEXUS firmware entry point - FreeRTOS task skeleton.
 *
 * Creates the main application tasks on the ESP32-S3:
 *   - vm_task:          Bytecode VM execution (1ms tick)
 *   - serial_task:      Wire protocol serial I/O
 *   - safety_task:      Safety state machine + watchdog
 *   - sensor_task:      Sensor polling and I/O acquisition
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"

static const char *TAG = "nexus_main";

/* Task priorities (lower number = lower priority) */
#define NEXUS_TASK_VM_PRIORITY       5
#define NEXUS_TASK_SERIAL_PRIORITY   8
#define NEXUS_TASK_SAFETY_PRIORITY  20
#define NEXUS_TASK_SENSOR_PRIORITY   6

/* Task stack sizes in words */
#define NEXUS_TASK_VM_STACK      4096
#define NEXUS_TASK_SERIAL_STACK  4096
#define NEXUS_TASK_SAFETY_STACK  4096
#define NEXUS_TASK_SENSOR_STACK  4096

/* ===================================================================
 * Task skeleton functions
 * =================================================================== */

static void vm_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "vm_task: started (priority=%d, stack=%u)",
             uxTaskPriorityGet(NULL), uxTaskGetStackHighWaterMark(NULL));

    for (;;) {
        /* TODO: Execute one VM tick (fetch-decode-execute loop) */
        vTaskDelay(pdMS_TO_TICKS(1)); /* 1ms tick */
    }
}

static void serial_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "serial_task: started (priority=%d, stack=%u)",
             uxTaskPriorityGet(NULL), uxTaskGetStackHighWaterMark(NULL));

    for (;;) {
        /* TODO: Process wire protocol frames via UART */
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

static void safety_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "safety_task: started (priority=%d, stack=%u)",
             uxTaskPriorityGet(NULL), uxTaskGetStackHighWaterMark(NULL));

    for (;;) {
        /* TODO: Run safety state machine, feed watchdog, monitor heartbeats */
        vTaskDelay(pdMS_TO_TICKS(10)); /* 10ms period */
    }
}

static void sensor_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "sensor_task: started (priority=%d, stack=%u)",
             uxTaskPriorityGet(NULL), uxTaskGetStackHighWaterMark(NULL));

    for (;;) {
        /* TODO: Poll sensors, populate VM sensor registers */
        vTaskDelay(pdMS_TO_TICKS(10)); /* 100Hz poll rate */
    }
}

/* ===================================================================
 * ESP-IDF entry point
 * =================================================================== */

void app_main(void) {
    ESP_LOGI(TAG, "=== NEXUS Runtime v0.1.0 ===");
    ESP_LOGI(TAG, "Target: ESP32-S3 (Xtensa LX7, 240MHz)");
    ESP_LOGI(TAG, "FreeRTOS tick rate: %d Hz", configTICK_RATE_HZ);

    /* Create FreeRTOS tasks */
    BaseType_t result;

    result = xTaskCreate(vm_task, "vm", NEXUS_TASK_VM_STACK,
                         NULL, NEXUS_TASK_VM_PRIORITY, NULL);
    if (result != pdPASS) {
        ESP_LOGE(TAG, "Failed to create vm_task");
    }

    result = xTaskCreate(serial_task, "serial", NEXUS_TASK_SERIAL_STACK,
                         NULL, NEXUS_TASK_SERIAL_PRIORITY, NULL);
    if (result != pdPASS) {
        ESP_LOGE(TAG, "Failed to create serial_task");
    }

    result = xTaskCreate(safety_task, "safety", NEXUS_TASK_SAFETY_STACK,
                         NULL, NEXUS_TASK_SAFETY_PRIORITY, NULL);
    if (result != pdPASS) {
        ESP_LOGE(TAG, "Failed to create safety_task");
    }

    result = xTaskCreate(sensor_task, "sensor", NEXUS_TASK_SENSOR_STACK,
                         NULL, NEXUS_TASK_SENSOR_PRIORITY, NULL);
    if (result != pdPASS) {
        ESP_LOGE(TAG, "Failed to create sensor_task");
    }

    ESP_LOGI(TAG, "All tasks created. NEXUS operational.");
}
