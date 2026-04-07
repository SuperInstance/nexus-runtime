/**
 * @file nexus_main.c
 * @brief NEXUS firmware entry point - FreeRTOS task wiring.
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
#include "driver/uart.h"

#include "vm.h"
#include "safety_sm.h"
#include "hal.h"
#include "frame.h"

static const char *TAG = "nexus_main";

/* ===================================================================
 * Global firmware state
 * =================================================================== */

static vm_state_t       g_vm;
static hal_context_t    g_hal;
static safety_sm_t      g_safety_sm;
static frame_receiver_t g_frame_rx;

/* UART config for serial task */
#define NEXUS_UART_NUM      UART_NUM_0
#define NEXUS_UART_BAUD     921600
#define NEXUS_UART_BUF_SIZE 1024
#define NEXUS_UART_RD_BUF   256

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
 * Task functions
 * =================================================================== */

static void vm_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "vm_task: started (priority=%d, stack=%u)",
             uxTaskPriorityGet(NULL), uxTaskGetStackHighWaterMark(NULL));

    for (;;) {
        vm_error_t err = vm_tick(&g_vm);
        if (err != VM_OK && err != ERR_CYCLE_BUDGET_EXCEEDED) {
            ESP_LOGW(TAG, "vm_tick error: %d", (int)err);
        }

        /* Drain actuator outputs to HAL after each tick */
        hal_drain_actuators(&g_hal);

        vTaskDelay(pdMS_TO_TICKS(1)); /* 1ms tick */
    }
}

static void serial_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "serial_task: started (priority=%d, stack=%u)",
             uxTaskPriorityGet(NULL), uxTaskGetStackHighWaterMark(NULL));

    uint8_t rx_byte;
    int bytes_read;

    for (;;) {
        bytes_read = uart_read_bytes(NEXUS_UART_NUM, &rx_byte, 1,
                                     pdMS_TO_TICKS(10));
        if (bytes_read > 0) {
            uint32_t frame_err = frame_receiver_feed(&g_frame_rx, rx_byte);
            if (frame_err == FRAME_ERR_NONE) {
                uint16_t frame_len = 0;
                const uint8_t *frame_data = frame_receiver_get_data(
                    &g_frame_rx, &frame_len);
                if (frame_data && frame_len > 0) {
                    ESP_LOGD(TAG, "serial: complete frame (%u bytes)",
                             (unsigned)frame_len);
                    /* TODO: Dispatch frame to wire protocol handler */
                }
            } else if (frame_err != 0) {
                g_frame_rx.error_count++;
            }
        }
    }
}

static void safety_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "safety_task: started (priority=%d, stack=%u)",
             uxTaskPriorityGet(NULL), uxTaskGetStackHighWaterMark(NULL));

    safety_sm_init(&g_safety_sm);

    for (;;) {
        uint32_t now = xTaskGetTickCount() * portTICK_PERIOD_MS;

        /* Run HAL-level safety checks (heartbeat, watchdog, overcurrent) */
        hal_update_safety_state(&g_hal, now);

        /* Feed hardware watchdog */
        hal_feed_watchdog(&g_hal);

        /* If HAL detected a safety event, forward to the state machine */
        if (g_hal.safety_state != g_safety_sm.current_state) {
            safety_event_t evt = SAFETY_EVT_BOOT_COMPLETE;
            if (g_hal.estop_triggered) {
                evt = SAFETY_EVT_ESTOP_TRIGGERED;
            } else {
                evt = SAFETY_EVT_SENSOR_FAILURE;
            }
            safety_sm_process_event(&g_safety_sm, evt, now);
            g_hal.safety_state = g_safety_sm.current_state;
            ESP_LOGW(TAG, "safety: transition to %s",
                     safety_sm_state_name(g_safety_sm.current_state));
        }

        vTaskDelay(pdMS_TO_TICKS(10)); /* 10ms period */
    }
}

static void sensor_task(void *pvParameters) {
    (void)pvParameters;
    ESP_LOGI(TAG, "sensor_task: started (priority=%d, stack=%u)",
             uxTaskPriorityGet(NULL), uxTaskGetStackHighWaterMark(NULL));

    for (;;) {
        uint32_t now = xTaskGetTickCount() * portTICK_PERIOD_MS;

        /* Update all sensor readings through HAL */
        hal_update_sensors(&g_hal, now);

        /* Push valid sensor readings into VM sensor registers */
        for (uint8_t i = 0; i < VM_SENSOR_COUNT; i++) {
            if (g_hal.sensors[i].valid) {
                vm_set_sensor(&g_vm, i, g_hal.sensors[i].value);
            }
        }

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

    /* Initialize subsystems */
    hal_init(&g_hal);
    vm_init(&g_vm, NULL, 0);  /* bytecode loaded later via OTA */
    frame_receiver_init(&g_frame_rx);

    /* Configure UART for serial wire protocol */
    uart_config_t uart_cfg = {
        .baud_rate = NEXUS_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_driver_install(NEXUS_UART_NUM, NEXUS_UART_BUF_SIZE,
                        NEXUS_UART_BUF_SIZE, 0, NULL, 0);
    uart_param_config(NEXUS_UART_NUM, &uart_cfg);

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
