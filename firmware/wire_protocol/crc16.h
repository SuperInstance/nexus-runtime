/**
 * @file crc16.h
 * @brief NEXUS Wire Protocol - CRC-16/CCITT-FALSE.
 *
 * Polynomial: 0x1021
 * Initial:     0xFFFF
 * Final XOR:   0x0000
 * Check value: 0x29B1 for "123456789"
 */

#ifndef NEXUS_CRC16_H
#define NEXUS_CRC16_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute CRC-16/CCITT-FALSE over a data buffer.
 * @param data Input data.
 * @param len Length of data in bytes.
 * @return 16-bit CRC value (big-endian ready).
 */
uint16_t crc16_ccitt(const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_CRC16_H */
