/**
 * @file sensor_bus.h
 * @brief NEXUS Drivers - Sensor bus abstraction (I2C/SPI/1-Wire).
 */

#ifndef NEXUS_SENSOR_BUS_H
#define NEXUS_SENSOR_BUS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SENSOR_BUS_MAX_DEVICES  16
#define SENSOR_READ_MAX_BYTES   32

typedef enum {
    SENSOR_BUS_I2C = 0,
    SENSOR_BUS_SPI = 1,
    SENSOR_BUS_ONEWIRE = 2,
} sensor_bus_type_t;

typedef struct {
    sensor_bus_type_t type;
    uint8_t address;
    bool initialized;
    uint32_t last_read_ms;
    uint32_t error_count;
} sensor_device_t;

/**
 * @brief Initialize the sensor bus.
 */
void sensor_bus_init(void);

/**
 * @brief Register a sensor device on the bus.
 * @param type Bus type (I2C, SPI, 1-Wire).
 * @param address Device address.
 * @return Device handle index, or -1 on error.
 */
int sensor_bus_register(sensor_bus_type_t type, uint8_t address);

/**
 * @brief Read data from a sensor device.
 * @param device_index Device handle.
 * @param reg Register to read.
 * @param data Output buffer.
 * @param len Number of bytes to read.
 * @return true on success, false on error.
 */
bool sensor_bus_read(uint8_t device_index, uint8_t reg,
                     uint8_t *data, uint16_t len);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_SENSOR_BUS_H */
