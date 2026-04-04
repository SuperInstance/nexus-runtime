/**
 * @file sensor_bus.c
 * @brief NEXUS Drivers - Sensor bus implementation (stub).
 */

#include "sensor_bus.h"
#include <string.h>

static sensor_device_t devices[SENSOR_BUS_MAX_DEVICES];
static bool bus_initialized = false;

void sensor_bus_init(void) {
    memset(devices, 0, sizeof(devices));
    bus_initialized = true;
}

int sensor_bus_register(sensor_bus_type_t type, uint8_t address) {
    if (!bus_initialized) {
        return -1;
    }
    for (int i = 0; i < SENSOR_BUS_MAX_DEVICES; i++) {
        if (!devices[i].initialized) {
            devices[i].type = type;
            devices[i].address = address;
            devices[i].initialized = true;
            devices[i].error_count = 0;
            return i;
        }
    }
    return -1;
}

bool sensor_bus_read(uint8_t device_index, uint8_t reg,
                     uint8_t *data, uint16_t len) {
    if (!bus_initialized || device_index >= SENSOR_BUS_MAX_DEVICES) {
        return false;
    }
    if (!devices[device_index].initialized || !data) {
        return false;
    }

    /* TODO: Implement actual I2C/SPI/1-Wire read */
    (void)reg;
    (void)len;
    memset(data, 0, len);
    return true;
}
