/* Internal layout shared by the Blink runtime translation units.
 * Not part of the public API. SPDX-License-Identifier: Apache-2.0 */
#ifndef BLINK_INTERNAL_H
#define BLINK_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "blink.h"

#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE && defined(BLINK_W8A8) && BLINK_W8A8
#error "BLINK_ACCELERATE and BLINK_W8A8 are alternative backends; pick one"
#endif

/* ------------------------------------------------------------- container */

#define BLINK_MAGIC "BLNKMDL"      /* 7 chars + NUL = 8 bytes */
/* Version 2 moved film.gamma and film.beta from int8 to fp32 and added the
 * cross layer. Adding a header field or a tensor is append-only and does not
 * need a bump -- a version-1 reader sees zeros in the reserved space and
 * behaves as the older model. Changing an existing tensor's dtype is not:
 * a version-1 container would be read as malformed rather than as old, so the
 * version carries the difference. */
#define BLINK_FORMAT_VERSION 3u
#define BLINK_FLAG_LITTLE_ENDIAN 1u
#define BLINK_HEADER_BYTES 128u
#define BLINK_TENSOR_ENTRY_BYTES 64u
#define BLINK_TENSOR_NAME_BYTES 32u
#define BLINK_ALIGN 64u

enum { BLINK_DTYPE_F32 = 0, BLINK_DTYPE_I8 = 1 };

/* Segment ids added to byte embeddings so the head can tell the three input
 * roles apart even though they share one encoder. */
enum { BLINK_SEG_STATE = 0, BLINK_SEG_QUESTION = 1, BLINK_SEG_OPTION = 2 };

#define BLINK_SEGMENTS 3u

/* On-disk header. Every field is little-endian; the loader refuses other
 * byte orders rather than swapping, so a mapped file is used in place. */
typedef struct blink_header {
    char magic[8];
    uint32_t format_version;
    uint32_t flags;
    uint32_t header_bytes;
    uint32_t tensor_count;
    uint64_t blob_offset;
    uint64_t blob_bytes;
    uint32_t crc32;
    uint32_t reserved0;
    /* configuration, 64 bytes */
    uint32_t width;
    uint32_t blocks;
    uint32_t ffn_width;
    uint32_t rank;
    uint32_t heads;
    uint32_t conv_width;
    uint32_t stride;
    uint32_t bigram_buckets;
    uint32_t max_state;
    uint32_t max_question;
    uint32_t max_option;
    float temperature;
    uint32_t vocab;
    uint32_t mixer_blocks;
    uint32_t film;
    uint32_t cross;
    uint32_t reserved1[1];
    char name[16];
} blink_header;

typedef struct blink_tensor_entry {
    char name[BLINK_TENSOR_NAME_BYTES];
    uint32_t dtype;
    uint32_t ndim;
    uint32_t dims[4];
    uint64_t offset; /* relative to blob_offset */
} blink_tensor_entry;

/* A resolved pointer into the mapped blob. Quantized tensors carry one fp32
 * scale per output row; `scale` is NULL for fp32 tensors.
 *
 * `dense` is NULL except in a BLINK_ACCELERATE build, where the projection
 * matrices are dequantized once at open time (row r is data[r] * scale[r],
 * row-major) so that they can be handed to cblas_sgemm. It lives in memory
 * the model owns, not in the mapped container. */
typedef struct blink_tensor {
    const void *data;
    const float *scale;
    uint32_t rows;
    uint32_t cols;
    uint32_t dtype;
    const float *dense;
    const int8_t *packed; /* BLINK_W8A8 with I8MM only: see blink_pack_i8mm */
} blink_tensor;

/* ---------------------------------------------------------------- weights */

/* The stem runs once at full byte resolution before pooling: a depthwise
 * convolution for local n-grams and a pointwise projection for channel
 * mixing. Keeping the expensive feed-forward blocks behind the pooling step is
 * what makes a byte-level encoder affordable. */
typedef struct blink_stem_weights {
    const float *ln_gain, *ln_bias;
    const float *conv;   /* [conv_width, width] fp32 */
    blink_tensor proj;   /* [width, width] i8        */
} blink_stem_weights;

/* Self-attention over the pooled positions. The encoder is otherwise linear in
 * the sequence length; this is the one content-addressed path that can bind two
 * distant positions to each other. It is affordable only because it runs after
 * the pool: at stride 8 a 512-byte state is 64 positions, so the quadratic term
 * is small next to the block feed-forward. */
typedef struct blink_mixer_weights {
    int present;
    const float *ln_gain, *ln_bias;
    blink_tensor q, k, v, o; /* [width, width] i8 */
} blink_mixer_weights;

/* One-directional attention from the question to the state.
 *
 * The state and the question are encoded as separate sequences, which is what
 * makes a cached state exactly reusable. The cost is that nothing inside the
 * encoder ever compares a question byte against a state byte: the two meet
 * only in the option head, which takes a weighted *average* over both, and an
 * average cannot express "are these the same word".
 *
 * This layer restores that comparison without giving up the cache. The
 * question attends to the state; the state never attends to the question, so
 * the state encoding stays a function of the state alone. Its keys and values
 * are computed once in blink_state_set and reused by every later question.
 */
typedef struct blink_cross_weights {
    int present;
    const float *ln_gain, *ln_bias;
    blink_tensor q, k, v, o; /* [width, width] i8 */
} blink_cross_weights;

typedef struct blink_block_weights {
    const float *ln1_gain, *ln1_bias;
    const float *conv;   /* [width, conv_width] fp32 */
    blink_tensor glob;   /* [width, width] i8        */
    const float *ln2_gain, *ln2_bias;
    blink_tensor fc1;    /* [ffn_width, width] i8    */
    blink_tensor fc2;    /* [width, ffn_width] i8    */
    blink_mixer_weights mixer;
} blink_block_weights;

struct blink_model {
    /* mapping bookkeeping */
    void *map_base;    /* non-NULL when the model owns an mmap              */
    size_t map_size;
    const uint8_t *base; /* start of the container (mapped or borrowed)     */
    size_t size;

    blink_header header;
    blink_model_info info;

    blink_tensor unigram;   /* [256, width] i8            */
    blink_tensor bigram;    /* [buckets, width] i8        */
    blink_tensor pos_state; /* [max_state, width] i8      */
    blink_tensor pos_quest; /* [max_question, width] i8   */
    blink_tensor pos_option;/* [max_option, width] i8     */
    const float *segment;   /* [3, width] f32             */

    blink_stem_weights stem;
    blink_cross_weights cross;
    blink_block_weights *blocks; /* owned, blocks entries */

    const float *ctx_ln_gain, *ctx_ln_bias;
    const float *opt_ln_gain, *opt_ln_bias;

    /* Feature-wise linear modulation of the option vectors by a summary of the
     * question. Without it an option's query carries no information about the
     * question, so no question-by-option interaction term can exist and any
     * task needing one is unlearnable. See docs/ARCHITECTURE.md. */
    const float *film_ln_gain, *film_ln_bias;
    const float *film_gamma, *film_beta; /* [width, width] f32 */
    blink_tensor wq, wk, wv; /* [rank, width] i8          */
    float logit_scale;

    /* BLINK_ACCELERATE only: one allocation holding every `dense` matrix.
     * NULL in the default build. */
    float *dense_pool;
    size_t dense_bytes;

    /* BLINK_W8A8 on a CPU with I8MM only: the projections rearranged into
     * the 2x8 blocks SMMLA reads, one allocation. NULL otherwise. */
    int8_t *packed_pool;
    size_t packed_bytes;
};

/* ---------------------------------------------------------------- session */

struct blink_session {
    const blink_model *model;
    blink_limits limits;
    void *owned;        /* non-NULL when created by blink_session_create   */

    uint32_t context_capacity; /* max_state + max_question                 */

    /* caches */
    int has_state;
    int has_menu;
    int has_score;
    uint32_t state_bytes;
    uint32_t question_bytes;
    uint32_t state_positions;    /* pooled */
    uint32_t question_positions; /* pooled */
    uint32_t menu_size;

    /* arena slices, all 64-byte aligned */
    float *x;        /* [seq_capacity, width]        encoder residual      */
    float *a;        /* [seq_capacity, width]        encoder scratch       */
    float *ffn;      /* [ffn_width]                                        */
    float *gmean;    /* [width]                                            */
    float *gproj;    /* [width]                                            */
    float *keys;     /* [context_capacity, rank]                           */
    float *values;   /* [context_capacity, rank]                           */
    float *optvec;   /* [max_options, width]   cached option encodings     */
    float *queries;  /* [max_options, rank]    rebuilt on every score       */
    float *attn;     /* [max_options, context_capacity] head-averaged      */
    float *scores;   /* [context_capacity]                                 */
    float *headbuf;  /* [rank]                                             */
    float *qhead;    /* [rank]   unit-length query slice                   */
    float *tmpw;     /* [width]                                            */
    float *qsum;     /* [width]   mean of the question's pooled states      */
    float *mixk;     /* [pool_capacity, width]  mixer keys                 */
    float *mixv;     /* [pool_capacity, width]  mixer values               */
    float *crossk;   /* [state_positions, width] cross keys, cached        */
    float *crossv;   /* [state_positions, width] cross values, cached      */
    float *logits;   /* [max_options]                                      */
    int8_t *qx;      /* BLINK_W8A8 only: [BLINK_ROW_BLOCK, widest input]
                        quantized activations for one group of rows        */
    uint32_t seq_capacity;
};

/* ---------------------------------------------------------------- kernels */

void blink_layernorm(float *restrict out, const float *restrict in,
                     const float *restrict gain, const float *restrict bias,
                     uint32_t width);

/* y[r] = scale[r] * sum_j w[r][j] * x[j] over an int8 weight matrix. */
void blink_matmul_i8(float *restrict out, const blink_tensor *w,
                     const float *restrict x);

/* Input vectors multiplied per weight-row load by blink_matmul_i8_rows. Four
 * vectors of four NEON accumulators each fill half the register file. */
#define BLINK_ROW_BLOCK 4u

/* Rows the runtime hands to blink_matmul_i8_rows per call, which also sizes
 * the per-group scratch in the session arena. The NEON kernel gains nothing
 * past BLINK_ROW_BLOCK; cblas_sgemm needs real matrices to reach the matrix
 * unit, so the Accelerate build passes up to 64 rows at a time. */
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
#define BLINK_GROUP_ROWS 64u
#else
#define BLINK_GROUP_ROWS BLINK_ROW_BLOCK
#endif

/* blink_matmul_i8 applied to `count` input vectors x + i * x_stride, writing
 * out + i * out_stride. Bit-identical to calling blink_matmul_i8 per vector;
 * it only converts each weight once per group of BLINK_ROW_BLOCK vectors.
 * The Accelerate build replaces it with one cblas_sgemm over the dense copy,
 * which is not bit-identical to the scalar definition (see blink_kernels.c). */
void blink_matmul_i8_rows(float *restrict out, size_t out_stride,
                          const blink_tensor *w, const float *restrict x,
                          size_t x_stride, uint32_t count);

/* ---------------------------------------------------- W8A8 (BLINK_W8A8=1)
 *
 * The same projections with the activations quantized too. Each input row x
 * gets its own symmetric scale: m = max|x_c|, q_c = round_half_even(x_c *
 * (127 / m)), clamped to [-127, 127], and s_x = m / 127 (all zero when m is
 * zero). The product is then an exact int32 sum,
 *
 *     out[r] = ((float)(sum_c w[r][c] * q_c) * scale[r]) * s_x,
 *
 * so the scalar, SDOT and I8MM kernels are bit-identical to each other: they
 * differ only in how they add integers. What differs from the default build is
 * the activation rounding itself, which is a change to the model, not to the
 * arithmetic, and is measured in docs/RESULTS.md. */

/* Quantize one row as above; returns s_x. */
float blink_quantize_row(int8_t *restrict q, const float *restrict x,
                         uint32_t n);

/* blink_matmul_i8_rows with W8A8 arithmetic. `scratch` holds
 * 2 * BLINK_ROW_BLOCK * w->cols bytes; any `count` is processed a group at a
 * time. */
void blink_matmul_w8a8_rows(float *restrict out, size_t out_stride,
                            const blink_tensor *w, const float *restrict x,
                            size_t x_stride, uint32_t count,
                            int8_t *restrict scratch);

/* Detect the optional int8 instructions once (called at model open) and
 * report which W8A8 kernel runs: 0 scalar, 1 SDOT, 2 I8MM. */
void blink_kernels_init(void);

/* Bytes of the SMMLA layout of a rows x cols matrix, and the rearrangement
 * itself: every block of four rows and sixteen columns is stored as the four
 * 16-byte operands (rows 0-1 low halves, rows 0-1 high halves, rows 2-3 low,
 * rows 2-3 high). Rows past a multiple of four and columns past a multiple
 * of sixteen are not packed; the kernel reads those from the original. */
size_t blink_packed_i8mm_bytes(uint32_t rows, uint32_t cols);
void blink_pack_i8mm(int8_t *restrict packed, const int8_t *restrict q,
                     uint32_t rows, uint32_t cols);
int blink_w8a8_kernel(void);
int blink_w8a8_force_kernel(int kernel);

/* Centred depthwise 1-D convolution with zero padding, accumulated into out. */
void blink_dwconv_accum(float *restrict out, const float *restrict in,
                        const float *restrict weight, uint32_t length,
                        uint32_t width, uint32_t kernel);

void blink_matmul_f32(float *restrict out, const float *restrict w,
                      const float *restrict x, uint32_t rows, uint32_t cols);

/* Mean-pool non-overlapping windows of `stride` positions, in place. The last
 * window may be short and is divided by its real count. Returns the number of
 * pooled positions. */
uint32_t blink_pool(float *restrict data, uint32_t length, uint32_t width,
                    uint32_t stride);

/* The scalar reference for one int8-by-fp32 row product, and whether the SIMD
 * path is compiled in. Both exist so the test suite can compare the paths. */
float blink_dot_i8_scalar(const int8_t *row, const float *x, uint32_t cols);
int blink_simd_enabled(void);

/* x86-64 only: which W8A32 kernel runs (0 scalar, 1 SSE2, 2 AVX2 + FMA), and
 * a way for the tests to choose one. Both return the level in use. */
int blink_x86_level(void);
int blink_x86_force_level(int level);
int blink_accelerate_enabled(void);

void blink_l2_normalize(float *restrict values, uint32_t count);

void blink_softmax(float *restrict values, uint32_t count);
float blink_dot(const float *restrict a, const float *restrict b, uint32_t n);

/* The two halves of one attention read, over `count` rows of a strided
 * matrix: the scaled dot products against a query, and the weighted sum of
 * value rows accumulated into `out`. */
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
/* One matrix-vector product each, in blink_kernels.c. */
void blink_attention_scores(float *restrict scores, const float *restrict keys,
                            size_t key_stride, const float *restrict query,
                            uint32_t dim, uint32_t count, float scale);
void blink_attention_accum(float *restrict out, const float *restrict values,
                           size_t value_stride, const float *restrict weights,
                           uint32_t dim, uint32_t count);
#else
/* The normative loops, inline so the default build compiles to exactly what
 * it did before they were factored out. */
static inline void blink_attention_scores(float *restrict scores,
                                          const float *restrict keys,
                                          size_t key_stride,
                                          const float *restrict query,
                                          uint32_t dim, uint32_t count,
                                          float scale)
{
    for (uint32_t l = 0; l < count; ++l) {
        scores[l] = blink_dot(query, keys + (size_t)l * key_stride, dim) * scale;
    }
}

static inline void blink_attention_accum(float *restrict out,
                                         const float *restrict values,
                                         size_t value_stride,
                                         const float *restrict weights,
                                         uint32_t dim, uint32_t count)
{
    for (uint32_t l = 0; l < count; ++l) {
        const float weight = weights[l];
        const float *restrict value = values + (size_t)l * value_stride;
        for (uint32_t c = 0; c < dim; ++c) {
            out[c] += weight * value[c];
        }
    }
}
#endif

uint32_t blink_bigram_index(uint8_t previous, uint8_t current,
                            uint32_t buckets);

uint32_t blink_crc32(uint32_t seed, const void *data, size_t size);

#endif /* BLINK_INTERNAL_H */
