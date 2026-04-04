/**
 * @file actuator_drv.c
 * @brief NEXUS Drivers - Actuator output driver (stub).
 */

#include "actuator_drv.h"
#include <string.h>

static actuator_channel_t channels[ACTUATOR_MAX_CHANNELS];
static bool drv_initialized = false;

void actuator_init(void) {
    memset(channels, 0, sizeof(channels));
    for (int i = 0; i < ACTUATOR_MAX_CHANNELS; i++) {
        channels[i].type = ACTUATOR_TYPE_LED;
        channels[i].current_value = 0.0f;
        channels[i].safe_value = 0.0f;
        channels[i].min_value = 0.0f;
        channels[i].max_value = 1.0f;
        channels[i].max_rate = 1.0f;
        channels[i].enabled = false;
    }
    drv_initialized = true;
}

bool actuator_set(uint8_t channel, float value) {
    if (!drv_initialized || channel >= ACTUATOR_MAX_CHANNELS) {
        return false;
    }
    if (!channels[channel].enabled) {
        return false;
    }

    /* TODO: Implement rate limiting and clamping */
    channels[channel].current_value = value;
    return true;
}

void actuator_force_safe(void) {
    for (int i = 0; i < ACTUATOR_MAX_CHANNELS; i++) {
        channels[i].current_value = channels[i].safe_value;
    }
}

float actuator_get(uint8_t channel) {
    if (!drv_initialized || channel >= ACTUATOR_MAX_CHANNELS) {
        return 0.0f;
    }
    return channels[channel].current_value;
}
