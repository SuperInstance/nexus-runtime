/**
 * @file cobs.h
 * @brief NEXUS Wire Protocol - COBS (Consistent Overhead Byte Stuffing).
 *
 * COBS encode/decode for zero-byte delimited framing.
 * Worst-case overhead: 1 byte per 254 bytes (0.4%).
 */

#ifndef NEXUS_COBS_H
#define NEXUS_COBS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief COBS encode a byte buffer.
 * @param src Source data (may contain zeros).
 * @param src_len Length of source data.
 * @param dst Destination buffer (must be >= src_len + (src_len/254) + 2).
 * @param dst_max Maximum destination buffer size.
 * @return Encoded length, or 0 on error.
 */
size_t cobs_encode(const uint8_t *src, size_t src_len,
                   uint8_t *dst, size_t dst_max);

/**
 * @brief COBS decode a byte buffer.
 * @param src COBS-encoded data.
 * @param src_len Length of encoded data.
 * @param dst Destination buffer (must be >= src_len).
 * @param dst_max Maximum destination buffer size.
 * @return Decoded length, or 0 on error.
 */
size_t cobs_decode(const uint8_t *src, size_t src_len,
                   uint8_t *dst, size_t dst_max);

#ifdef __cplusplus
}
#endif

#endif /* NEXUS_COBS_H */
