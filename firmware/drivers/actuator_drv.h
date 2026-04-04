/**
 * @file actuator_drv.h
 * @brief NEXUS Drivers - Actuator output driver.
 *
 * Manages actuator outputs with safety constraints:
 *   - All outputs start at safe state on boot
 *   - Rate limiting on all actuator changes
 *   - Safe state values defined per actuator type
 */

#ifndef NEXUS_ACTUATOR_DRV_H
#define NEXUS_ACTUATOR_DRV_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ACTUATOR_MAX_CHANNELS 16

typedef enum {
    ACTUATOR_TYPE_SERVO = 0,
    ACTUATOR_TYPE_RELAY = 1,
    ACTUATOR_TYPE_MOTOR_PWM = 2,
    ACTUATOR_TYPE_SOLENOID = 3,
    ACTUATOR_TYPE_LED = 4,
    ACTUATOR_TYPE_BUZZER = 5,
} actuator_type_t;

typedef struct {
    actuator_type_t type;
    float current_value;
    float safe_value;
    float min_value;
    float max_value;
    float max_rate;
    uint32_t last_write_ms;
    bool enabled;
} actuator_channel_t;

/**
 * @brief Initialize actuator driver (all outputs to safe state).
 */
void actuator_init(void);

/**
 * @brief Set an actuator channel value.
 * @param channel Channel index.
 * @param value Desired value (float32).
 * @return true if write succeeded, false if rejected (safety/rate limit).
 */
bool actuator_set(uint8_t channel, float value);

/**
 * @brief Force all actuators to safe state.
 */
void actuator_force_safe(void);

/**
 * @brief Get current actuator value.
 * @param channel Channel index.
 * @return Current value, or safe_value if channel invalid.
 */
float actuator_get(uint8_t channel);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_ACTUATOR_DRV_H */
