/* blink_synth -- write a valid .blink container filled with pseudo-random
 * weights.
 *
 * The C unit tests and the benchmarks use this so that neither depends on
 * Python or on a trained checkpoint. A synthesised model produces meaningless
 * decisions but exercises exactly the same code paths and the same shapes as a
 * trained one, which is what a latency or memory measurement needs.
 *
 *   blink_synth OUT.blink [preset]
 *
 * Presets: nano, tiny, small. Output is byte-for-byte reproducible.
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blink_internal.h"

typedef struct geometry {
    const char *name;
    uint32_t width, blocks, ffn_width, rank, heads, conv_width, stride, buckets;
    uint32_t max_state, max_question, max_option, mixer_blocks, film, cross;
} geometry;

/* Kept in step with python/blink_train/config.py PRESETS. */
static const geometry PRESETS[] = {
    {"blink-nano", 16, 2, 32, 16, 1, 3, 2, 64, 48, 24, 12, 1, 1, 1},
    {"blink-tiny", 64, 2, 128, 64, 2, 5, 4, 4096, 256, 128, 48, 1, 1, 1},
    {"blink-small", 192, 4, 384, 192, 4, 5, 8, 32768, 512, 192, 64, 2, 1, 1},
};

/* ------------------------------------------------------- deterministic RNG */

static uint64_t rng_state = 0x243F6A8885A308D3ull;

static uint32_t next_u32(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

/* Uniform in [-1, 1). */
static float next_unit(void)
{
    return (float)((double)next_u32() / 2147483648.0 - 1.0);
}

/* ------------------------------------------------------------ tensor table */

typedef struct pending {
    char name[BLINK_TENSOR_NAME_BYTES];
    uint32_t dtype, ndim, dims[4];
    void *data;
    size_t bytes;
} pending;

#define MAX_TENSORS 1024
static pending tensors[MAX_TENSORS];
static uint32_t tensor_count = 0;

static void push(const char *name, uint32_t dtype, uint32_t rows, uint32_t cols,
                 void *data, size_t bytes)
{
    if (tensor_count >= MAX_TENSORS) {
        fprintf(stderr, "too many tensors\n");
        exit(1);
    }
    pending *entry = &tensors[tensor_count++];
    memset(entry, 0, sizeof *entry);
    strncpy(entry->name, name, BLINK_TENSOR_NAME_BYTES - 1);
    entry->dtype = dtype;
    entry->ndim = cols ? 2u : 1u;
    entry->dims[0] = rows;
    entry->dims[1] = cols;
    entry->data = data;
    entry->bytes = bytes;
}

static void add_f32(const char *name, uint32_t rows, uint32_t cols, float spread)
{
    const size_t count = (size_t)rows * (cols ? cols : 1u);
    float *values = (float *)malloc(count * sizeof *values);
    for (size_t i = 0; i < count; ++i) {
        values[i] = next_unit() * spread;
    }
    push(name, BLINK_DTYPE_F32, rows, cols, values, count * sizeof *values);
}

static void add_ones(const char *name, uint32_t rows, float value)
{
    float *values = (float *)malloc((size_t)rows * sizeof *values);
    for (uint32_t i = 0; i < rows; ++i) {
        values[i] = value;
    }
    push(name, BLINK_DTYPE_F32, rows, 0u, values, (size_t)rows * sizeof *values);
}

static void add_i8(const char *name, uint32_t rows, uint32_t cols, float spread)
{
    const size_t count = (size_t)rows * cols;
    int8_t *quantized = (int8_t *)malloc(count);
    for (size_t i = 0; i < count; ++i) {
        quantized[i] = (int8_t)((int32_t)(next_u32() % 255u) - 127);
    }
    push(name, BLINK_DTYPE_I8, rows, cols, quantized, count);

    char scale_name[BLINK_TENSOR_NAME_BYTES];
    snprintf(scale_name, sizeof scale_name, "%s.scale", name);
    float *scale = (float *)malloc((size_t)rows * sizeof *scale);
    for (uint32_t r = 0; r < rows; ++r) {
        scale[r] = spread / 127.0f;
    }
    push(scale_name, BLINK_DTYPE_F32, rows, 0u, scale,
         (size_t)rows * sizeof *scale);
}

/* ----------------------------------------------------------------- writing */

static void write_u32(uint8_t *base, size_t offset, uint32_t value)
{
    base[offset] = (uint8_t)(value & 0xFFu);
    base[offset + 1] = (uint8_t)((value >> 8) & 0xFFu);
    base[offset + 2] = (uint8_t)((value >> 16) & 0xFFu);
    base[offset + 3] = (uint8_t)((value >> 24) & 0xFFu);
}

static void write_u64(uint8_t *base, size_t offset, uint64_t value)
{
    write_u32(base, offset, (uint32_t)(value & 0xFFFFFFFFu));
    write_u32(base, offset + 4, (uint32_t)(value >> 32));
}

static void write_f32(uint8_t *base, size_t offset, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    write_u32(base, offset, bits);
}

static size_t align_up(size_t value)
{
    return (value + BLINK_ALIGN - 1u) & ~((size_t)BLINK_ALIGN - 1u);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s OUT.blink [nano|tiny|small]\n", argv[0]);
        return 2;
    }
    const char *want = argc > 2 ? argv[2] : "tiny";
    const geometry *g = NULL;
    for (size_t i = 0; i < sizeof PRESETS / sizeof PRESETS[0]; ++i) {
        if (strcmp(PRESETS[i].name + 6, want) == 0) {
            g = &PRESETS[i];
        }
    }
    if (!g) {
        fprintf(stderr, "unknown preset %s\n", want);
        return 2;
    }

    const uint32_t w = g->width;
    add_i8("emb.unigram", 256u, w, 0.6f);
    add_i8("emb.bigram", g->buckets, w, 0.6f);
    add_i8("emb.pos_state", g->max_state, w, 0.3f);
    add_i8("emb.pos_question", g->max_question, w, 0.3f);
    add_i8("emb.pos_option", g->max_option, w, 0.3f);
    add_f32("emb.segment", BLINK_SEGMENTS, w, 0.2f);

    add_ones("stem.ln.gain", w, 1.0f);
    add_ones("stem.ln.bias", w, 0.0f);
    add_f32("stem.conv", g->conv_width, w, 0.4f);
    add_i8("stem.proj", w, w, 0.1f);

    for (uint32_t b = 0; b < g->blocks; ++b) {
        char name[BLINK_TENSOR_NAME_BYTES];
        snprintf(name, sizeof name, "block%u.ln1.gain", b);
        add_ones(name, w, 1.0f);
        snprintf(name, sizeof name, "block%u.ln1.bias", b);
        add_ones(name, w, 0.0f);
        snprintf(name, sizeof name, "block%u.conv", b);
        add_f32(name, g->conv_width, w, 0.4f);
        snprintf(name, sizeof name, "block%u.glob", b);
        add_i8(name, w, w, 0.1f);
        snprintf(name, sizeof name, "block%u.ln2.gain", b);
        add_ones(name, w, 1.0f);
        snprintf(name, sizeof name, "block%u.ln2.bias", b);
        add_ones(name, w, 0.0f);
        snprintf(name, sizeof name, "block%u.fc1", b);
        add_i8(name, g->ffn_width, w, 0.25f);
        snprintf(name, sizeof name, "block%u.fc2", b);
        add_i8(name, w, g->ffn_width, 0.25f);

        if (b >= g->blocks - g->mixer_blocks) {
            snprintf(name, sizeof name, "block%u.mix.ln.gain", b);
            add_ones(name, w, 1.0f);
            snprintf(name, sizeof name, "block%u.mix.ln.bias", b);
            add_ones(name, w, 0.0f);
            const char *parts[4] = {"q", "k", "v", "o"};
            for (int i = 0; i < 4; ++i) {
                snprintf(name, sizeof name, "block%u.mix.%s", b, parts[i]);
                add_i8(name, w, w, 0.2f);
            }
        }
    }

    if (g->cross) {
        add_ones("cross.ln.gain", w, 1.0f);
        add_ones("cross.ln.bias", w, 0.0f);
        const char *cparts[4] = {"q", "k", "v", "o"};
        for (int i = 0; i < 4; ++i) {
            char cname[BLINK_TENSOR_NAME_BYTES];
            snprintf(cname, sizeof cname, "cross.%s", cparts[i]);
            add_i8(cname, w, w, 0.2f);
        }
    }

    if (g->film) {
        add_ones("film.ln.gain", w, 1.0f);
        add_ones("film.ln.bias", w, 0.0f);
        add_f32("film.gamma", w, w, 0.1f);
        add_f32("film.beta", w, w, 0.1f);
    }

    add_ones("head.ctx_ln.gain", w, 1.0f);
    add_ones("head.ctx_ln.bias", w, 0.0f);
    add_ones("head.opt_ln.gain", w, 1.0f);
    add_ones("head.opt_ln.bias", w, 0.0f);
    add_i8("head.q", g->rank, w, 0.3f);
    add_i8("head.k", g->rank, w, 0.3f);
    add_i8("head.v", g->rank, w, 0.3f);
    add_ones("head.logit_scale", 1u, 1.0f);

    /* lay out the blob */
    size_t blob_bytes = 0;
    size_t *offsets = (size_t *)malloc(tensor_count * sizeof *offsets);
    for (uint32_t i = 0; i < tensor_count; ++i) {
        offsets[i] = align_up(blob_bytes);
        blob_bytes = offsets[i] + tensors[i].bytes;
    }
    const size_t table_bytes = (size_t)tensor_count * BLINK_TENSOR_ENTRY_BYTES;
    const size_t blob_offset = align_up(BLINK_HEADER_BYTES + table_bytes);
    const size_t total = blob_offset + blob_bytes;

    uint8_t *file = (uint8_t *)calloc(1, total);
    memcpy(file, BLINK_MAGIC, 8);
    write_u32(file, 8, BLINK_FORMAT_VERSION);
    write_u32(file, 12, BLINK_FLAG_LITTLE_ENDIAN);
    write_u32(file, 16, BLINK_HEADER_BYTES);
    write_u32(file, 20, tensor_count);
    write_u64(file, 24, blob_offset);
    write_u64(file, 32, blob_bytes);
    write_u32(file, 40, 0u);
    write_u32(file, 48, w);
    write_u32(file, 52, g->blocks);
    write_u32(file, 56, g->ffn_width);
    write_u32(file, 60, g->rank);
    write_u32(file, 64, g->heads);
    write_u32(file, 68, g->conv_width);
    write_u32(file, 72, g->stride);
    write_u32(file, 76, g->buckets);
    write_u32(file, 80, g->max_state);
    write_u32(file, 84, g->max_question);
    write_u32(file, 88, g->max_option);
    write_f32(file, 92, 1.0f);
    write_u32(file, 96, 256u);
    write_u32(file, 100, g->mixer_blocks);
    write_u32(file, 104, g->film);
    write_u32(file, 108, g->cross);
    memcpy(file + 112, g->name, strlen(g->name));

    for (uint32_t i = 0; i < tensor_count; ++i) {
        uint8_t *entry = file + BLINK_HEADER_BYTES +
                         (size_t)i * BLINK_TENSOR_ENTRY_BYTES;
        memcpy(entry, tensors[i].name, strlen(tensors[i].name));
        write_u32(entry, 32, tensors[i].dtype);
        write_u32(entry, 36, tensors[i].ndim);
        for (int d = 0; d < 4; ++d) {
            write_u32(entry, 40u + (size_t)d * 4u, tensors[i].dims[d]);
        }
        write_u64(entry, 56, offsets[i]);
        memcpy(file + blob_offset + offsets[i], tensors[i].data, tensors[i].bytes);
        free(tensors[i].data);
    }

    write_u32(file, 40, blink_crc32(0u, file, total));

    FILE *out = fopen(argv[1], "wb");
    if (!out) {
        perror("fopen");
        return 1;
    }
    const size_t written = fwrite(file, 1, total, out);
    fclose(out);
    free(file);
    free(offsets);
    if (written != total) {
        fprintf(stderr, "short write\n");
        return 1;
    }
    printf("{\"path\":\"%s\",\"preset\":\"%s\",\"bytes\":%zu,\"tensors\":%u}\n",
           argv[1], g->name, total, tensor_count);
    return 0;
}
