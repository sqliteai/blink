/* Container parsing and weight resolution.
 *
 * A .blink file is mapped read-only and used in place: no weight byte is ever
 * copied into the heap. The only heap allocation a model makes is the small
 * per-block pointer table.
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* posix_memalign, clock_gettime and mmap are POSIX, not C99. glibc hides
 * them under -std=c99 unless a POSIX level is requested before any header;
 * Darwin exposes them anyway, and defining the macro there would hide other
 * interfaces (bench_memory's mach headers among them), so it is Linux only. */
#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "blink_internal.h"

/* ------------------------------------------------------------- utilities */

static uint32_t read_u32(const uint8_t *base, size_t offset)
{
    return (uint32_t)base[offset] | ((uint32_t)base[offset + 1] << 8) |
           ((uint32_t)base[offset + 2] << 16) |
           ((uint32_t)base[offset + 3] << 24);
}

static uint64_t read_u64(const uint8_t *base, size_t offset)
{
    return (uint64_t)read_u32(base, offset) |
           ((uint64_t)read_u32(base, offset + 4) << 32);
}

static float read_f32(const uint8_t *base, size_t offset)
{
    const uint32_t bits = read_u32(base, offset);
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static int is_power_of_two(uint32_t value)
{
    return value != 0u && (value & (value - 1u)) == 0u;
}

static size_t dtype_bytes(uint32_t dtype)
{
    return dtype == BLINK_DTYPE_I8 ? 1u : 4u;
}

static void fail(blink_status *status, blink_status value)
{
    if (status) {
        *status = value;
    }
}

/* -------------------------------------------------------- header parsing */

static int parse_header(const uint8_t *base, size_t size, blink_header *out)
{
    if (size < BLINK_HEADER_BYTES) {
        return 0;
    }
    memcpy(out->magic, base, 8);
    if (memcmp(out->magic, BLINK_MAGIC, 8) != 0) {
        return 0;
    }
    out->format_version = read_u32(base, 8);
    out->flags = read_u32(base, 12);
    out->header_bytes = read_u32(base, 16);
    out->tensor_count = read_u32(base, 20);
    out->blob_offset = read_u64(base, 24);
    out->blob_bytes = read_u64(base, 32);
    out->crc32 = read_u32(base, 40);
    out->reserved0 = read_u32(base, 44);
    out->width = read_u32(base, 48);
    out->blocks = read_u32(base, 52);
    out->ffn_width = read_u32(base, 56);
    out->rank = read_u32(base, 60);
    out->heads = read_u32(base, 64);
    out->conv_width = read_u32(base, 68);
    out->stride = read_u32(base, 72);
    out->bigram_buckets = read_u32(base, 76);
    out->max_state = read_u32(base, 80);
    out->max_question = read_u32(base, 84);
    out->max_option = read_u32(base, 88);
    out->temperature = read_f32(base, 92);
    out->vocab = read_u32(base, 96);
    out->mixer_blocks = read_u32(base, 100);
    out->film = read_u32(base, 104);
    out->cross = read_u32(base, 108);
    memcpy(out->name, base + 112, 16);
    out->name[15] = '\0';
    return 1;
}

static int header_is_sane(const blink_header *h, size_t size)
{
    if (h->header_bytes != BLINK_HEADER_BYTES) return 0;
    if (h->tensor_count == 0u || h->tensor_count > 4096u) return 0;
    if (h->width == 0u || h->width > 4096u || (h->width % 4u) != 0u) return 0;
    if (h->blocks == 0u || h->blocks > 64u) return 0;
    if (h->ffn_width == 0u || h->ffn_width > 16384u) return 0;
    if (h->rank == 0u || h->rank > 4096u) return 0;
    if (h->heads == 0u || h->heads > 64u || (h->rank % h->heads) != 0u) return 0;
    if ((h->conv_width & 1u) == 0u || h->conv_width > 31u) return 0;
    if (h->stride == 0u || h->stride > 64u) return 0;
    if (h->mixer_blocks > h->blocks) return 0;
    if (h->film > 1u) return 0;
    if (h->cross > 1u) return 0;
    if (h->cross && (h->width % h->heads) != 0u) return 0;
    if (h->mixer_blocks && (h->width % h->heads) != 0u) return 0;
    if (!is_power_of_two(h->bigram_buckets)) return 0;
    if (h->vocab != 256u) return 0;
    if (h->max_state == 0u || h->max_state > BLINK_MAX_SEQ) return 0;
    if (h->max_question == 0u || h->max_question > BLINK_MAX_SEQ) return 0;
    if (h->max_option == 0u || h->max_option > BLINK_MAX_SEQ) return 0;
    if (!(h->temperature > 0.0f) || !(h->temperature < 1000.0f)) return 0;

    const uint64_t table = (uint64_t)h->tensor_count * BLINK_TENSOR_ENTRY_BYTES;
    if (h->blob_offset < BLINK_HEADER_BYTES + table) return 0;
    if (h->blob_offset % BLINK_ALIGN != 0u) return 0;
    if (h->blob_offset > size) return 0;
    if (h->blob_bytes > size - h->blob_offset) return 0;
    return 1;
}

/* CRC covers the whole file with the checksum field itself read as zero. */
static uint32_t container_crc(const uint8_t *base, size_t size)
{
    static const uint8_t zeros[4] = {0, 0, 0, 0};
    uint32_t crc = blink_crc32(0u, base, 40u);
    crc = blink_crc32(crc, zeros, 4u);
    return blink_crc32(crc, base + 44u, size - 44u);
}

/* --------------------------------------------------------- tensor lookup */

static int find_tensor(const blink_model *model, const char *name,
                       blink_tensor_entry *out)
{
    const uint8_t *table = model->base + BLINK_HEADER_BYTES;
    for (uint32_t i = 0; i < model->header.tensor_count; ++i) {
        const uint8_t *entry = table + (size_t)i * BLINK_TENSOR_ENTRY_BYTES;
        if (strncmp((const char *)entry, name, BLINK_TENSOR_NAME_BYTES) == 0) {
            memcpy(out->name, entry, BLINK_TENSOR_NAME_BYTES);
            out->name[BLINK_TENSOR_NAME_BYTES - 1] = '\0';
            out->dtype = read_u32(entry, 32);
            out->ndim = read_u32(entry, 36);
            for (int d = 0; d < 4; ++d) {
                out->dims[d] = read_u32(entry, 40u + (size_t)d * 4u);
            }
            out->offset = read_u64(entry, 56);
            return 1;
        }
    }
    return 0;
}

static const void *resolve(const blink_model *model,
                           const blink_tensor_entry *entry, size_t elements)
{
    const size_t bytes = elements * dtype_bytes(entry->dtype);
    if (entry->offset % BLINK_ALIGN != 0u) return NULL;
    if (entry->offset > model->header.blob_bytes) return NULL;
    if (bytes > model->header.blob_bytes - entry->offset) return NULL;
    return model->base + model->header.blob_offset + entry->offset;
}

/* Fetch a fp32 tensor of exactly `rows x cols` elements. */
static const float *get_f32(const blink_model *model, const char *name,
                            uint32_t rows, uint32_t cols)
{
    blink_tensor_entry entry;
    if (!find_tensor(model, name, &entry)) return NULL;
    if (entry.dtype != BLINK_DTYPE_F32) return NULL;
    if (entry.dims[0] != rows || (cols != 0u && entry.dims[1] != cols)) return NULL;
    const size_t elements = (size_t)rows * (cols ? cols : 1u);
    return (const float *)resolve(model, &entry, elements);
}

/* Fetch an int8 matrix plus its companion "<name>.scale" fp32 row vector. */
static int get_i8(const blink_model *model, const char *name, uint32_t rows,
                  uint32_t cols, blink_tensor *out)
{
    blink_tensor_entry entry;
    char scale_name[BLINK_TENSOR_NAME_BYTES];
    if (!find_tensor(model, name, &entry)) return 0;
    if (entry.dtype != BLINK_DTYPE_I8 || entry.ndim != 2u) return 0;
    if (entry.dims[0] != rows || entry.dims[1] != cols) return 0;

    const void *data = resolve(model, &entry, (size_t)rows * cols);
    if (!data) return 0;

    if (snprintf(scale_name, sizeof scale_name, "%s.scale", name) >=
        (int)sizeof scale_name) {
        return 0;
    }
    const float *scale = get_f32(model, scale_name, rows, 0u);
    if (!scale) return 0;

    out->data = data;
    out->scale = scale;
    out->rows = rows;
    out->cols = cols;
    out->dtype = BLINK_DTYPE_I8;
    return 1;
}

/* ------------------------------------------------------------- weight map */

static uint64_t count_parameters(const blink_model *model)
{
    const blink_header *h = &model->header;
    uint64_t total = 0;
    total += (uint64_t)h->vocab * h->width;
    total += (uint64_t)h->bigram_buckets * h->width;
    total += (uint64_t)h->max_state * h->width;
    total += (uint64_t)h->max_question * h->width;
    total += (uint64_t)h->max_option * h->width;
    total += (uint64_t)BLINK_SEGMENTS * h->width;
    total += (uint64_t)2u * h->width +                   /* stem layernorm */
             (uint64_t)h->conv_width * h->width +        /* stem conv      */
             (uint64_t)h->width * h->width;              /* stem proj      */
    total += (uint64_t)h->blocks *
             ((uint64_t)2u * h->width +                  /* ln1            */
              (uint64_t)h->conv_width * h->width +       /* depthwise conv */
              (uint64_t)h->width * h->width +            /* global proj    */
              (uint64_t)2u * h->width +                  /* ln2            */
              (uint64_t)h->ffn_width * h->width * 2u);   /* fc1 + fc2      */
    total += (uint64_t)h->mixer_blocks *
             ((uint64_t)2u * h->width +                  /* mixer layernorm*/
              (uint64_t)4u * h->width * h->width);       /* q, k, v, o     */
    total += (uint64_t)h->film *
             ((uint64_t)2u * h->width +                  /* film layernorm */
              (uint64_t)2u * h->width * h->width);       /* gamma, beta    */
    total += (uint64_t)h->cross *
             ((uint64_t)2u * h->width +                  /* cross layernorm*/
              (uint64_t)4u * h->width * h->width);       /* q, k, v, o     */
    total += (uint64_t)4u * h->width;                    /* head layernorms*/
    total += (uint64_t)3u * h->rank * h->width;          /* q, k, v        */
    total += 1u;                                         /* logit scale    */
    return total;
}

static int bind_weights(blink_model *model)
{
    const blink_header *h = &model->header;
    const uint32_t width = h->width;

    if (!get_i8(model, "emb.unigram", h->vocab, width, &model->unigram)) return 0;
    if (!get_i8(model, "emb.bigram", h->bigram_buckets, width, &model->bigram)) return 0;
    if (!get_i8(model, "emb.pos_state", h->max_state, width, &model->pos_state)) return 0;
    if (!get_i8(model, "emb.pos_question", h->max_question, width, &model->pos_quest)) return 0;
    if (!get_i8(model, "emb.pos_option", h->max_option, width, &model->pos_option)) return 0;

    model->segment = get_f32(model, "emb.segment", BLINK_SEGMENTS, width);
    if (!model->segment) return 0;

    model->stem.ln_gain = get_f32(model, "stem.ln.gain", width, 0u);
    model->stem.ln_bias = get_f32(model, "stem.ln.bias", width, 0u);
    model->stem.conv = get_f32(model, "stem.conv", h->conv_width, width);
    if (!model->stem.ln_gain || !model->stem.ln_bias || !model->stem.conv) return 0;
    if (!get_i8(model, "stem.proj", width, width, &model->stem.proj)) return 0;

    model->cross.present = (int)h->cross;
    if (h->cross) {
        model->cross.ln_gain = get_f32(model, "cross.ln.gain", width, 0u);
        model->cross.ln_bias = get_f32(model, "cross.ln.bias", width, 0u);
        if (!model->cross.ln_gain || !model->cross.ln_bias) return 0;
        if (!get_i8(model, "cross.q", width, width, &model->cross.q)) return 0;
        if (!get_i8(model, "cross.k", width, width, &model->cross.k)) return 0;
        if (!get_i8(model, "cross.v", width, width, &model->cross.v)) return 0;
        if (!get_i8(model, "cross.o", width, width, &model->cross.o)) return 0;
    }

    model->blocks = (blink_block_weights *)calloc(h->blocks,
                                                  sizeof *model->blocks);
    if (!model->blocks) return 0;

    for (uint32_t b = 0; b < h->blocks; ++b) {
        blink_block_weights *bw = &model->blocks[b];
        char name[BLINK_TENSOR_NAME_BYTES];

#define BLINK_NAME(suffix)                                                     \
    (snprintf(name, sizeof name, "block%u." suffix, b) < (int)sizeof name      \
         ? name                                                                \
         : NULL)

        if (!BLINK_NAME("ln1.gain") || !(bw->ln1_gain = get_f32(model, name, width, 0u))) return 0;
        if (!BLINK_NAME("ln1.bias") || !(bw->ln1_bias = get_f32(model, name, width, 0u))) return 0;
        if (!BLINK_NAME("conv") || !(bw->conv = get_f32(model, name, h->conv_width, width))) return 0;
        if (!BLINK_NAME("glob") || !get_i8(model, name, width, width, &bw->glob)) return 0;
        if (!BLINK_NAME("ln2.gain") || !(bw->ln2_gain = get_f32(model, name, width, 0u))) return 0;
        if (!BLINK_NAME("ln2.bias") || !(bw->ln2_bias = get_f32(model, name, width, 0u))) return 0;
        if (!BLINK_NAME("fc1") || !get_i8(model, name, h->ffn_width, width, &bw->fc1)) return 0;
        if (!BLINK_NAME("fc2") || !get_i8(model, name, width, h->ffn_width, &bw->fc2)) return 0;

        bw->mixer.present = (b >= h->blocks - h->mixer_blocks);
        if (bw->mixer.present) {
            if (!BLINK_NAME("mix.ln.gain") ||
                !(bw->mixer.ln_gain = get_f32(model, name, width, 0u))) return 0;
            if (!BLINK_NAME("mix.ln.bias") ||
                !(bw->mixer.ln_bias = get_f32(model, name, width, 0u))) return 0;
            if (!BLINK_NAME("mix.q") || !get_i8(model, name, width, width, &bw->mixer.q)) return 0;
            if (!BLINK_NAME("mix.k") || !get_i8(model, name, width, width, &bw->mixer.k)) return 0;
            if (!BLINK_NAME("mix.v") || !get_i8(model, name, width, width, &bw->mixer.v)) return 0;
            if (!BLINK_NAME("mix.o") || !get_i8(model, name, width, width, &bw->mixer.o)) return 0;
        }
#undef BLINK_NAME
    }

    model->ctx_ln_gain = get_f32(model, "head.ctx_ln.gain", width, 0u);
    model->ctx_ln_bias = get_f32(model, "head.ctx_ln.bias", width, 0u);
    model->opt_ln_gain = get_f32(model, "head.opt_ln.gain", width, 0u);
    model->opt_ln_bias = get_f32(model, "head.opt_ln.bias", width, 0u);
    if (!model->ctx_ln_gain || !model->ctx_ln_bias || !model->opt_ln_gain ||
        !model->opt_ln_bias) {
        return 0;
    }
    if (h->film) {
        model->film_ln_gain = get_f32(model, "film.ln.gain", width, 0u);
        model->film_ln_bias = get_f32(model, "film.ln.bias", width, 0u);
        if (!model->film_ln_gain || !model->film_ln_bias) return 0;
        model->film_gamma = get_f32(model, "film.gamma", width, width);
        model->film_beta = get_f32(model, "film.beta", width, width);
        if (!model->film_gamma || !model->film_beta) return 0;
    }
    if (!get_i8(model, "head.q", h->rank, width, &model->wq)) return 0;
    if (!get_i8(model, "head.k", h->rank, width, &model->wk)) return 0;
    if (!get_i8(model, "head.v", h->rank, width, &model->wv)) return 0;

    const float *scale = get_f32(model, "head.logit_scale", 1u, 0u);
    if (!scale) return 0;
    model->logit_scale = scale[0];
    return 1;
}

#if (defined(BLINK_ACCELERATE) && BLINK_ACCELERATE) || \
    (defined(BLINK_W8A8) && BLINK_W8A8)
/* Every int8 projection the encoder and the head multiply by, i.e. every
 * tensor blink_matmul_i8 and blink_matmul_i8_rows see. The embedding tables
 * are read a row at a time and stay int8. Returns the number written. */
static uint32_t projections(blink_model *model, blink_tensor **out)
{
    uint32_t n = 0;
    out[n++] = &model->stem.proj;
    if (model->cross.present) {
        out[n++] = &model->cross.q;
        out[n++] = &model->cross.k;
        out[n++] = &model->cross.v;
        out[n++] = &model->cross.o;
    }
    for (uint32_t b = 0; b < model->header.blocks; ++b) {
        blink_block_weights *bw = &model->blocks[b];
        out[n++] = &bw->glob;
        out[n++] = &bw->fc1;
        out[n++] = &bw->fc2;
        if (bw->mixer.present) {
            out[n++] = &bw->mixer.q;
            out[n++] = &bw->mixer.k;
            out[n++] = &bw->mixer.v;
            out[n++] = &bw->mixer.o;
        }
    }
    out[n++] = &model->wq;
    out[n++] = &model->wk;
    out[n++] = &model->wv;
    return n;
}

#endif

#if defined(BLINK_W8A8) && BLINK_W8A8
/* Rearrange the projections once into the SMMLA layout, so the I8MM kernel
 * loads its weight operands instead of shuffling them on every product. Only
 * when that kernel is the one that runs; the mapped int8 weights stay in
 * place and are still what the other kernels, and the unpacked edges, read. */
static int build_packed(blink_model *model)
{
    if (blink_w8a8_kernel() != 2) {
        return 1;
    }
    blink_tensor *list[1u + 4u + 7u * 64u + 3u];
    const uint32_t count = projections(model, list);
    size_t bytes = 0;
    for (uint32_t i = 0; i < count; ++i) {
        bytes += blink_packed_i8mm_bytes(list[i]->rows, list[i]->cols);
    }
    if (bytes == 0u) {
        return 1;
    }
    model->packed_pool = (int8_t *)malloc(bytes);
    if (!model->packed_pool) return 0;
    model->packed_bytes = bytes;
    int8_t *cursor = model->packed_pool;
    for (uint32_t i = 0; i < count; ++i) {
        blink_tensor *t = list[i];
        blink_pack_i8mm(cursor, (const int8_t *)t->data, t->rows, t->cols);
        t->packed = cursor;
        cursor += blink_packed_i8mm_bytes(t->rows, t->cols);
    }
    return 1;
}
#endif

#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
/* Dequantize the projections once, into one allocation the model owns, so
 * cblas_sgemm can use them. Row r becomes data[r][c] * scale[r]: the scale is
 * folded into the weights instead of applied after the sum. The container
 * stays mapped and untouched; this is extra memory, reported in dense_bytes. */
static int build_dense(blink_model *model)
{
    /* stem, cross (4), 7 per block, head (3) */
    blink_tensor *list[1u + 4u + 7u * 64u + 3u];
    const uint32_t count = projections(model, list);

    size_t floats = 0;
    for (uint32_t i = 0; i < count; ++i) {
        floats += (size_t)list[i]->rows * list[i]->cols;
    }
    model->dense_pool = (float *)malloc(floats * sizeof(float));
    if (!model->dense_pool) return 0;
    model->dense_bytes = floats * sizeof(float);

    float *cursor = model->dense_pool;
    for (uint32_t i = 0; i < count; ++i) {
        blink_tensor *t = list[i];
        const int8_t *q = (const int8_t *)t->data;
        for (uint32_t r = 0; r < t->rows; ++r) {
            for (uint32_t c = 0; c < t->cols; ++c) {
                cursor[(size_t)r * t->cols + c] =
                    (float)q[(size_t)r * t->cols + c] * t->scale[r];
            }
        }
        t->dense = cursor;
        cursor += (size_t)t->rows * t->cols;
    }
    return 1;
}
#endif

static void fill_info(blink_model *model)
{
    const blink_header *h = &model->header;
    blink_model_info *info = &model->info;
    info->width = h->width;
    info->blocks = h->blocks;
    info->ffn_width = h->ffn_width;
    info->rank = h->rank;
    info->heads = h->heads;
    info->conv_width = h->conv_width;
    info->stride = h->stride;
    info->mixer_blocks = h->mixer_blocks;
    info->film = h->film;
    info->cross = h->cross;
    info->bigram_buckets = h->bigram_buckets;
    info->max_state = h->max_state;
    info->max_question = h->max_question;
    info->max_option = h->max_option;
    info->temperature = h->temperature;
    info->parameters = count_parameters(model);
    info->weights_bytes = h->blob_bytes;
    memset(info->name, 0, sizeof info->name);
    memcpy(info->name, h->name, 15);
}

/* ------------------------------------------------------------------- open */

static blink_model *open_common(const uint8_t *base, size_t size,
                                void *map_base, size_t map_size,
                                int verify_checksum, blink_status *status)
{
    blink_header header;
    if (!parse_header(base, size, &header)) {
        fail(status, BLINK_E_FORMAT);
        return NULL;
    }
    if (header.format_version != BLINK_FORMAT_VERSION) {
        fail(status, BLINK_E_VERSION);
        return NULL;
    }
    if ((header.flags & BLINK_FLAG_LITTLE_ENDIAN) == 0u) {
        fail(status, BLINK_E_FORMAT);
        return NULL;
    }
    if (!header_is_sane(&header, size)) {
        fail(status, BLINK_E_FORMAT);
        return NULL;
    }
    if (verify_checksum && container_crc(base, size) != header.crc32) {
        fail(status, BLINK_E_CHECKSUM);
        return NULL;
    }

    blink_model *model = (blink_model *)calloc(1, sizeof *model);
    if (!model) {
        fail(status, BLINK_E_IO);
        return NULL;
    }
    model->map_base = map_base;
    model->map_size = map_size;
    model->base = base;
    model->size = size;
    model->header = header;

    if (!bind_weights(model)) {
        free(model->blocks);
        free(model);
        fail(status, BLINK_E_FORMAT);
        return NULL;
    }
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
    if (!build_dense(model)) {
        free(model->blocks);
        free(model);
        fail(status, BLINK_E_IO);
        return NULL;
    }
#endif
    blink_kernels_init();
#if defined(BLINK_W8A8) && BLINK_W8A8
    if (!build_packed(model)) {
        free(model->blocks);
        free(model);
        fail(status, BLINK_E_IO);
        return NULL;
    }
#endif
    fill_info(model);
    fail(status, BLINK_OK);
    return model;
}

blink_model *blink_model_open_memory(const void *data, size_t size,
                                     int verify_checksum, blink_status *status)
{
    if (!data || size < BLINK_HEADER_BYTES) {
        fail(status, BLINK_E_INVALID);
        return NULL;
    }
    return open_common((const uint8_t *)data, size, NULL, 0, verify_checksum,
                       status);
}

blink_model *blink_model_open_file(const char *path, int verify_checksum,
                                   blink_status *status)
{
    if (!path) {
        fail(status, BLINK_E_INVALID);
        return NULL;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fail(status, BLINK_E_IO);
        return NULL;
    }
    struct stat info;
    if (fstat(fd, &info) != 0 || info.st_size < (off_t)BLINK_HEADER_BYTES) {
        close(fd);
        fail(status, BLINK_E_IO);
        return NULL;
    }
    const size_t size = (size_t)info.st_size;
    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fail(status, BLINK_E_IO);
        return NULL;
    }
    blink_model *model = open_common((const uint8_t *)map, size, map, size,
                                     verify_checksum, status);
    if (!model) {
        munmap(map, size);
    }
    return model;
}

void blink_model_close(blink_model *model)
{
    if (!model) {
        return;
    }
    free(model->dense_pool);
    free(model->packed_pool);
    free(model->blocks);
    if (model->map_base) {
        munmap(model->map_base, model->map_size);
    }
    free(model);
}

void blink_model_get_info(const blink_model *model, blink_model_info *out)
{
    if (model && out) {
        *out = model->info;
    }
}

const char *blink_status_string(blink_status status)
{
    switch (status) {
    case BLINK_OK: return "ok";
    case BLINK_E_INVALID: return "invalid argument";
    case BLINK_E_IO: return "io error";
    case BLINK_E_FORMAT: return "malformed container";
    case BLINK_E_VERSION: return "unsupported container version";
    case BLINK_E_CHECKSUM: return "checksum mismatch";
    case BLINK_E_NOSPACE: return "arena too small";
    case BLINK_E_LIMIT: return "input exceeds session limits";
    case BLINK_E_STATE: return "state or menu not set";
    }
    return "unknown";
}

const char *blink_version(void) { return BLINK_VERSION_STRING; }
