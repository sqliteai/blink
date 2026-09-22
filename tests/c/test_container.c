/* Container validation: a malformed file must be rejected, never trusted.
 * SPDX-License-Identifier: Apache-2.0 */

#include "blink_internal.h"
#include "harness.h"

static uint8_t *slurp(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(file, 0, SEEK_END);
    const long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc((size_t)length);
    if (fread(data, 1, (size_t)length, file) != (size_t)length) {
        exit(2);
    }
    fclose(file);
    *size = (size_t)length;
    return data;
}

/* Corrupt one field, confirm the loader refuses, restore it. */
static void expect_rejected(uint8_t *data, size_t size, size_t offset,
                            uint32_t replacement, blink_status expected,
                            const char *what)
{
    uint8_t original[4];
    memcpy(original, data + offset, 4);
    for (int i = 0; i < 4; ++i) {
        data[offset + (size_t)i] = (uint8_t)((replacement >> (8 * i)) & 0xFFu);
    }
    blink_status status = BLINK_OK;
    blink_model *model = blink_model_open_memory(data, size, 1, &status);
    ++blink_test_checks;
    if (model != NULL || status != expected) {
        ++blink_test_failures;
        fprintf(stderr, "corrupting %s was accepted (status %d, expected %d)\n",
                what, (int)status, (int)expected);
        blink_model_close(model);
    }
    memcpy(data + offset, original, 4);
}

int main(int argc, char **argv)
{
    const char *path = model_path(argc, argv, BLINK_BUILD_DIR "/synth-nano.blink");
    size_t size = 0;
    uint8_t *data = slurp(path, &size);

    /* the untouched file loads, with a checked checksum */
    blink_status status = BLINK_E_IO;
    blink_model *model = blink_model_open_memory(data, size, 1, &status);
    CHECK_EQ(status, BLINK_OK);
    CHECK(model != NULL);
    if (!model) {
        return 1;
    }

    blink_model_info info;
    blink_model_get_info(model, &info);
    CHECK(info.width > 0u);
    CHECK(info.rank % info.heads == 0u);
    CHECK((info.bigram_buckets & (info.bigram_buckets - 1u)) == 0u);
    CHECK(info.conv_width % 2u == 1u);
    CHECK(info.stride >= 1u);
    CHECK(info.parameters > 0u);
    CHECK(info.weights_bytes > 0u);
    CHECK(info.weights_bytes < size);
    CHECK(info.temperature > 0.0f);
    CHECK_EQ(strncmp(info.name, "blink-", 6), 0);
    blink_model_close(model);

    /* and every corruption is caught */
    expect_rejected(data, size, 0, 0x4B4E4C42u ^ 0xFFu, BLINK_E_FORMAT, "magic");
    expect_rejected(data, size, 8, 999u, BLINK_E_VERSION, "format version");
    /* An older container must be reported as old, not as corrupt: a reader
     * that says "malformed" sends you looking for a damaged file. */
    expect_rejected(data, size, 8, 1u, BLINK_E_VERSION, "a previous version");
    expect_rejected(data, size, 12, 0u, BLINK_E_FORMAT, "endianness flag");
    expect_rejected(data, size, 16, 64u, BLINK_E_FORMAT, "header size");
    expect_rejected(data, size, 20, 0u, BLINK_E_FORMAT, "tensor count");
    expect_rejected(data, size, 24, 7u, BLINK_E_FORMAT, "blob offset alignment");
    expect_rejected(data, size, 24, 0xFFFFFF00u, BLINK_E_FORMAT, "blob offset range");
    expect_rejected(data, size, 32, 0xFFFFFF00u, BLINK_E_FORMAT, "blob size");
    expect_rejected(data, size, 40, 0xDEADBEEFu, BLINK_E_CHECKSUM, "checksum");
    expect_rejected(data, size, 48, 7u, BLINK_E_FORMAT, "width not a multiple of 4");
    expect_rejected(data, size, 52, 0u, BLINK_E_FORMAT, "zero blocks");
    expect_rejected(data, size, 64, 0u, BLINK_E_FORMAT, "zero heads");
    expect_rejected(data, size, 64, 5u, BLINK_E_FORMAT, "heads not dividing rank");
    expect_rejected(data, size, 68, 4u, BLINK_E_FORMAT, "even convolution width");
    expect_rejected(data, size, 72, 0u, BLINK_E_FORMAT, "zero stride");
    expect_rejected(data, size, 76, 100u, BLINK_E_FORMAT, "non power-of-two buckets");
    expect_rejected(data, size, 80, 0u, BLINK_E_FORMAT, "zero state capacity");
    expect_rejected(data, size, 92, 0u, BLINK_E_FORMAT, "zero temperature");
    expect_rejected(data, size, 96, 128u, BLINK_E_FORMAT, "wrong vocabulary");

    /* a truncated file is rejected, at every truncation point */
    for (size_t cut = 1; cut < size; cut = cut * 2u + 1u) {
        status = BLINK_OK;
        blink_model *partial = blink_model_open_memory(data, cut, 1, &status);
        CHECK(partial == NULL);
        CHECK(status != BLINK_OK);
        blink_model_close(partial);
    }

    /* a corrupted weight byte is caught by the checksum but not without it */
    const size_t tail = size - 1u;
    data[tail] ^= 0xFFu;
    status = BLINK_OK;
    CHECK(blink_model_open_memory(data, size, 1, &status) == NULL);
    CHECK_EQ(status, BLINK_E_CHECKSUM);
    blink_model *unchecked = blink_model_open_memory(data, size, 0, &status);
    CHECK_EQ(status, BLINK_OK);
    CHECK(unchecked != NULL);
    blink_model_close(unchecked);
    data[tail] ^= 0xFFu;

    /* null and undersized inputs */
    CHECK(blink_model_open_memory(NULL, size, 1, &status) == NULL);
    CHECK_EQ(status, BLINK_E_INVALID);
    CHECK(blink_model_open_memory(data, 8, 1, &status) == NULL);
    CHECK_EQ(status, BLINK_E_INVALID);
    CHECK(blink_model_open_file("/nonexistent/blink.model", 1, &status) == NULL);
    CHECK_EQ(status, BLINK_E_IO);

    /* mapping the same file gives the same geometry as the memory path */
    blink_model *mapped = blink_model_open_file(path, 1, &status);
    CHECK_EQ(status, BLINK_OK);
    CHECK(mapped != NULL);
    if (mapped) {
        blink_model_info mapped_info;
        blink_model_get_info(mapped, &mapped_info);
        CHECK_EQ(memcmp(&mapped_info, &info, sizeof info), 0);
        blink_model_close(mapped);
    }

    /* closing a null model is a no-op rather than a crash */
    blink_model_close(NULL);
    blink_model_get_info(NULL, NULL);
    CHECK(strlen(blink_status_string(BLINK_E_CHECKSUM)) > 0);
    /* the library reports the version the header declares */
    CHECK(strcmp(blink_version(), BLINK_VERSION_STRING) == 0);
    CHECK(strcmp(BLINK_VERSION_STRING, "0.1.0") == 0);
    CHECK_EQ(BLINK_VERSION_NUMBER, 100);
#if BLINK_VERSION_NUMBER != BLINK_VERSION_MAJOR * 10000 + BLINK_VERSION_MINOR * 100 + BLINK_VERSION_PATCH
#error "BLINK_VERSION_NUMBER must be usable in #if"
#endif

    free(data);
    TEST_MAIN_END();
}
