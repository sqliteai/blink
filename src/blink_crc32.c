/* CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320) over the container.
 * SPDX-License-Identifier: Apache-2.0 */

#include "blink_internal.h"

static uint32_t table[256];
static int table_ready = 0;

static void build_table(void)
{
    for (uint32_t i = 0; i < 256u; ++i) {
        uint32_t c = i;
        for (int bit = 0; bit < 8; ++bit) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        table[i] = c;
    }
    table_ready = 1;
}

uint32_t blink_crc32(uint32_t seed, const void *data, size_t size)
{
    if (!table_ready) {
        build_table();
    }
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t crc = seed ^ 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ bytes[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}
