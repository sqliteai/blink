/* Session arena, encoder and option-attention head.
 *
 * Nothing in this file allocates after blink_session_init returns. The whole
 * working set is one contiguous arena whose size is a pure function of the
 * model and the declared limits.
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* posix_memalign, clock_gettime and mmap are POSIX, not C99. glibc hides
 * them under -std=c99 unless a POSIX level is requested before any header;
 * Darwin exposes them anyway, and defining the macro there would hide other
 * interfaces (bench_memory's mach headers among them), so it is Linux only. */
#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <string.h>
#include <time.h>

#include "blink_internal.h"

/* ---------------------------------------------------------------- timing */

static double monotonic_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ----------------------------------------------------------- bump arena */

typedef struct arena {
    uint8_t *base;
    size_t size;
    size_t used;
    int overflow;
} arena;

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

/* When base is NULL the arena only measures, which is how
 * blink_session_size and blink_session_init stay in step by construction. */
static void *arena_take(arena *a, size_t bytes)
{
    a->used = align_up(a->used, BLINK_ALIGN);
    const size_t offset = a->used;
    a->used += bytes;
    if (a->base == NULL) {
        return NULL;
    }
    if (a->used > a->size) {
        a->overflow = 1;
        return NULL;
    }
    return a->base + offset;
}

static float *arena_floats(arena *a, size_t count)
{
    return (float *)arena_take(a, count * sizeof(float));
}

static void resolve_limits(const blink_model *model, const blink_limits *in,
                           blink_limits *out)
{
    const blink_model_info *info = &model->info;
    out->max_state = (in && in->max_state) ? in->max_state : info->max_state;
    out->max_question =
        (in && in->max_question) ? in->max_question : info->max_question;
    out->max_option = (in && in->max_option) ? in->max_option : info->max_option;
    out->max_options = (in && in->max_options) ? in->max_options : 16u;

    if (out->max_state > info->max_state) out->max_state = info->max_state;
    if (out->max_question > info->max_question) out->max_question = info->max_question;
    if (out->max_option > info->max_option) out->max_option = info->max_option;
    if (out->max_options > BLINK_MAX_OPTIONS) out->max_options = BLINK_MAX_OPTIONS;
    if (out->max_options < 2u) out->max_options = 2u;
}

/* Lays out (and, when a->base is set, binds) every session buffer. */
static void layout(arena *a, blink_session *session, const blink_model *model,
                   const blink_limits *limits)
{
    const uint32_t width = model->info.width;
    const uint32_t rank = model->info.rank;

    uint32_t sequence = limits->max_state;
    if (limits->max_question > sequence) sequence = limits->max_question;
    if (limits->max_option > sequence) sequence = limits->max_option;

    /* The head attends over pooled positions, so the key/value cache and the
     * attention buffer shrink with the stride even though the encoder scratch
     * still has to hold the full byte sequence. */
    const uint32_t stride = model->info.stride;
    const uint32_t context = (limits->max_state + stride - 1u) / stride +
                             (limits->max_question + stride - 1u) / stride;

    blink_session *s = session;
    if (s) {
        s->model = model;
        s->limits = *limits;
        s->seq_capacity = sequence;
        s->context_capacity = context;
        s->has_state = s->has_menu = s->has_score = 0;
        s->state_bytes = s->question_bytes = 0;
        s->state_positions = s->question_positions = s->menu_size = 0;
        s->owned = NULL;
    }

#define BIND(field, count)                                                     \
    do {                                                                       \
        float *p = arena_floats(a, (count));                                   \
        if (s) s->field = p;                                                   \
    } while (0)

    /* `a` also stages BLINK_GROUP_ROWS layer-normed option vectors at a time
     * in build_queries, so it never has fewer rows than that, and the
     * row-sized scratch buffers hold one group of rows for
     * blink_matmul_i8_rows. */
    const uint32_t scratch_rows =
        sequence > BLINK_GROUP_ROWS ? sequence : BLINK_GROUP_ROWS;
    BIND(x, (size_t)sequence * width);
    BIND(a, (size_t)scratch_rows * width);
    BIND(ffn, (size_t)BLINK_GROUP_ROWS * model->info.ffn_width);
    BIND(gmean, (size_t)BLINK_GROUP_ROWS * width);
    BIND(gproj, (size_t)BLINK_GROUP_ROWS * width);
    BIND(keys, (size_t)context * rank);
    BIND(values, (size_t)context * rank);
    BIND(optvec, (size_t)limits->max_options * width);
    BIND(queries, (size_t)limits->max_options * rank);
    BIND(attn, (size_t)limits->max_options * context);
    BIND(scores, context);
    BIND(headbuf, rank);
    BIND(qhead, rank);
    BIND(logits, limits->max_options);
    BIND(tmpw, (size_t)BLINK_GROUP_ROWS * width);
    BIND(qsum, width);

    /* The mixer caches one key and one value row per pooled position. When no
     * block carries a mixer the two slices are empty and cost nothing. */
    const uint32_t pooled_capacity =
        model->info.mixer_blocks ? (sequence + stride - 1u) / stride : 0u;
    BIND(mixk, (size_t)pooled_capacity * width);
    BIND(mixv, (size_t)pooled_capacity * width);

    /* The question-to-state cross layer keeps one key and one value row per
     * pooled *state* position, filled once by blink_state_set. */
    const uint32_t cross_capacity =
        model->info.cross ? (limits->max_state + stride - 1u) / stride : 0u;
    BIND(crossk, (size_t)cross_capacity * width);
    BIND(crossv, (size_t)cross_capacity * width);
#undef BIND

#if defined(BLINK_W8A8) && BLINK_W8A8
    /* One group of quantized input rows, plus the same rows rearranged for
     * SMMLA; the widest projection input is the feed-forward's. */
    {
        const uint32_t widest =
            model->info.ffn_width > width ? model->info.ffn_width : width;
        int8_t *q = (int8_t *)arena_take(a, (size_t)2u * BLINK_ROW_BLOCK * widest);
        if (s) s->qx = q;
    }
#else
    if (s) s->qx = NULL;
#endif
}

size_t blink_session_size(const blink_model *model, const blink_limits *limits)
{
    if (!model) {
        return 0;
    }
    blink_limits resolved;
    resolve_limits(model, limits, &resolved);

    arena measure = {NULL, 0, 0, 0};
    (void)arena_take(&measure, sizeof(blink_session));
    layout(&measure, NULL, model, &resolved);
    return align_up(measure.used, BLINK_ALIGN);
}

blink_session *blink_session_init(void *memory, size_t size,
                                  const blink_model *model,
                                  const blink_limits *limits,
                                  blink_status *status)
{
    if (!memory || !model || ((uintptr_t)memory & 15u) != 0u) {
        if (status) *status = BLINK_E_INVALID;
        return NULL;
    }
    blink_limits resolved;
    resolve_limits(model, limits, &resolved);
    if (size < blink_session_size(model, &resolved)) {
        if (status) *status = BLINK_E_NOSPACE;
        return NULL;
    }

    arena a = {(uint8_t *)memory, size, 0, 0};
    blink_session *session = (blink_session *)arena_take(&a, sizeof *session);
    memset(session, 0, sizeof *session);
    layout(&a, session, model, &resolved);
    if (a.overflow) {
        if (status) *status = BLINK_E_NOSPACE;
        return NULL;
    }
    if (status) *status = BLINK_OK;
    return session;
}

/* --------------------------------------------------------------- encoder */

/* Every projection in the encoder and the head goes through here, so the
 * W8A8 build changes one function rather than every call site. */
static void project(blink_session *s, float *restrict out, size_t out_stride,
                    const blink_tensor *w, const float *restrict x,
                    size_t x_stride, uint32_t count)
{
#if defined(BLINK_W8A8) && BLINK_W8A8
    blink_matmul_w8a8_rows(out, out_stride, w, x, x_stride, count, s->qx);
#else
    (void)s;
    blink_matmul_i8_rows(out, out_stride, w, x, x_stride, count);
#endif
}

static void embed_row(float *restrict out, const blink_tensor *table,
                      uint32_t row, uint32_t width, int accumulate)
{
    const int8_t *restrict q = (const int8_t *)table->data + (size_t)row * width;
    const float scale = table->scale[row];
    if (accumulate) {
        for (uint32_t c = 0; c < width; ++c) {
            out[c] += (float)q[c] * scale;
        }
    } else {
        for (uint32_t c = 0; c < width; ++c) {
            out[c] = (float)q[c] * scale;
        }
    }
}

/* Several byte sequences encoded side by side in one pass.
 *
 * Every operation in the encoder is either position-wise (embeddings, layer
 * norms, projections, the feed-forward) or confined to one sequence (the
 * depthwise convolutions, the pool, the global mean, the mixer's attention).
 * Packing independent sequences next to each other in `x` and honouring those
 * boundaries is therefore the same computation as encoding them one at a
 * time -- bit for bit, since no sum changes its terms or their order (in the
Accelerate build, to fp32 rounding: cblas_sgemm may round a row differently
depending on how many rows share the call) -- while
 * the position-wise projections now see full groups of BLINK_GROUP_ROWS rows
 * instead of one short sequence's worth. blink_menu_set packs its options
 * this way and blink_score_batch its questions.
 *
 * `start` and `length` are in pooled positions: after encoding, sequence i
 * occupies rows start[i] .. start[i] + length[i] - 1 of `x`, contiguously. */
typedef struct blink_segments {
    uint32_t count;
    uint32_t start[BLINK_MAX_OPTIONS];
    uint32_t length[BLINK_MAX_OPTIONS];
} blink_segments;

/* How many of `lengths[0..count)` fit in one pass: the bytes must fit the
 * encoder scratch and the pooled positions must fit `pooled_room`. The first
 * item always fits, because every item is already within the session limits. */
static uint32_t segments_that_fit(const blink_session *s, const size_t *lengths,
                                  uint32_t count, uint32_t pooled_room)
{
    const uint32_t stride = s->model->info.stride;
    const uint32_t mixer_room = (s->seq_capacity + stride - 1u) / stride;
    if (pooled_room > mixer_room) {
        pooled_room = mixer_room;
    }
    uint32_t bytes = 0u, pooled = 0u, taken = 0u;
    while (taken < count && taken < BLINK_MAX_OPTIONS) {
        const uint32_t length = (uint32_t)lengths[taken];
        const uint32_t positions = (length + stride - 1u) / stride;
        if (taken > 0u && (bytes + length > s->seq_capacity ||
                           pooled + positions > pooled_room)) {
            break;
        }
        bytes += length;
        pooled += positions;
        ++taken;
    }
    return taken;
}

/* Self-attention over pooled positions, accumulated into `x`. Each position
 * attends only within its own sequence.
 *
 * Only the keys and the values are materialised for the whole pass; queries,
 * attended vectors and output projections need one group of rows of scratch. */
static void mix_positions(blink_session *s, const blink_mixer_weights *mixer,
                          const blink_segments *seg, uint32_t total)
{
    const blink_model *m = s->model;
    const uint32_t width = m->info.width;
    const uint32_t heads = m->info.heads;
    const uint32_t head_dim = width / heads;
    const float inverse = 1.0f / sqrtf((float)head_dim);

    for (uint32_t t = 0; t < total; ++t) {
        blink_layernorm(s->a + (size_t)t * width, s->x + (size_t)t * width,
                        mixer->ln_gain, mixer->ln_bias, width);
    }
    project(s, s->mixk, width, &mixer->k, s->a, width, total);
    project(s, s->mixv, width, &mixer->v, s->a, width, total);

    /* Queries and output projections run one group of rows at a time. The
     * attention for each position reads only the normalised inputs and the
     * cached keys and values, never `x`, so updating `x` group by group is
     * the same computation as updating it position by position. */
    uint32_t owner = 0u;
    for (uint32_t t0 = 0; t0 < total; t0 += BLINK_GROUP_ROWS) {
        const uint32_t group =
            total - t0 < BLINK_GROUP_ROWS ? total - t0 : BLINK_GROUP_ROWS;
        project(s, s->gmean, width, &mixer->q,
                             s->a + (size_t)t0 * width, width, group);
        memset(s->tmpw, 0, (size_t)group * width * sizeof(float));
        for (uint32_t g = 0; g < group; ++g) {
            while (t0 + g >= seg->start[owner] + seg->length[owner]) {
                ++owner;
            }
            const uint32_t first = seg->start[owner];
            const uint32_t length = seg->length[owner];
            const float *keys = s->mixk + (size_t)first * width;
            const float *values = s->mixv + (size_t)first * width;
            const float *query = s->gmean + (size_t)g * width;
            float *attended = s->tmpw + (size_t)g * width;
            for (uint32_t h = 0; h < heads; ++h) {
                const uint32_t base = h * head_dim;
                blink_attention_scores(s->scores, keys + base, width,
                                       query + base, head_dim, length, inverse);
                blink_softmax(s->scores, length);
                blink_attention_accum(attended + base, values + base, width,
                                      s->scores, head_dim, length);
            }
        }
        project(s, s->gproj, width, &mixer->o, s->tmpw, width, group);
        for (uint32_t g = 0; g < group; ++g) {
            float *xt = s->x + (size_t)(t0 + g) * width;
            const float *projected = s->gproj + (size_t)g * width;
            for (uint32_t c = 0; c < width; ++c) {
                xt[c] += projected[c];
            }
        }
    }
}
/* Attend from `length` positions in session->x to the cached state keys and
 * values, accumulating into x. Mirrors mix_positions, but the keys and values
 * come from another sequence and were computed once. */
static void cross_attend(blink_session *s, uint32_t length, uint32_t targets)
{
    const blink_model *m = s->model;
    const blink_cross_weights *cross = &m->cross;
    const uint32_t width = m->info.width;
    const uint32_t heads = m->info.heads;
    const uint32_t head_dim = width / heads;
    const float inverse = 1.0f / sqrtf((float)head_dim);

    for (uint32_t t = 0; t < length; ++t) {
        blink_layernorm(s->a + (size_t)t * width, s->x + (size_t)t * width,
                        cross->ln_gain, cross->ln_bias, width);
    }
    for (uint32_t t0 = 0; t0 < length; t0 += BLINK_GROUP_ROWS) {
        const uint32_t group =
            length - t0 < BLINK_GROUP_ROWS ? length - t0 : BLINK_GROUP_ROWS;
        project(s, s->gmean, width, &cross->q,
                             s->a + (size_t)t0 * width, width, group);
        memset(s->tmpw, 0, (size_t)group * width * sizeof(float));
        for (uint32_t g = 0; g < group; ++g) {
            const float *query = s->gmean + (size_t)g * width;
            float *attended = s->tmpw + (size_t)g * width;
            for (uint32_t h = 0; h < heads; ++h) {
                const uint32_t base = h * head_dim;
                blink_attention_scores(s->scores, s->crossk + base, width,
                                       query + base, head_dim, targets,
                                       inverse);
                blink_softmax(s->scores, targets);
                blink_attention_accum(attended + base, s->crossv + base, width,
                                      s->scores, head_dim, targets);
            }
        }
        project(s, s->gproj, width, &cross->o, s->tmpw, width, group);
        for (uint32_t g = 0; g < group; ++g) {
            float *xt = s->x + (size_t)(t0 + g) * width;
            const float *projected = s->gproj + (size_t)g * width;
            for (uint32_t c = 0; c < width; ++c) {
                xt[c] += projected[c];
            }
        }
    }
}

/* Project the encoded state into the cross layer's key and value cache. The
 * state never attends to anything, so this depends on the state alone and a
 * cached state stays exactly reusable. */
static void write_cross_kv(blink_session *s, uint32_t length)
{
    const blink_model *m = s->model;
    const uint32_t width = m->info.width;
    for (uint32_t t = 0; t < length; ++t) {
        blink_layernorm(s->a + (size_t)t * width, s->x + (size_t)t * width,
                        m->cross.ln_gain, m->cross.ln_bias, width);
    }
    project(s, s->crossk, width, &m->cross.k, s->a, width, length);
    project(s, s->crossv, width, &m->cross.v, s->a, width, length);
}

static float relu2(float value)
{
    return value > 0.0f ? value * value : 0.0f;
}

/* Encode `count` byte sequences side by side into session->x and describe
 * where each one ended up in `seg`. Returns the total number of pooled
 * positions; sequence i contributes ceil(lengths[i] / stride) of them. The
 * caller has already checked the pass fits (segments_that_fit).
 *
 * The shape of the pass is: byte embeddings, one full-resolution stem for
 * local n-grams, a mean pool by `stride`, then the feed-forward blocks at the
 * reduced resolution. Putting the pool before the blocks is what keeps a
 * byte-level encoder affordable: the expensive part runs on a quarter or an
 * eighth as many positions. */
static uint32_t encode_sequences(blink_session *s, const char *const *texts,
                                 const size_t *lengths, uint32_t count,
                                 uint32_t segment,
                                 const blink_tensor *positions,
                                 blink_segments *seg)
{
    const blink_model *m = s->model;
    const uint32_t width = m->info.width;
    const uint32_t stride = m->info.stride;

    /* byte embeddings, each sequence from its own position zero */
    uint32_t bytes_total = 0u;
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t *bytes = (const uint8_t *)texts[i];
        const uint32_t length = (uint32_t)lengths[i];
        for (uint32_t t = 0; t < length; ++t) {
            float *xt = s->x + (size_t)(bytes_total + t) * width;
            const uint8_t current = bytes[t];
            const uint8_t previous = (t == 0u) ? 0u : bytes[t - 1u];
            embed_row(xt, &m->unigram, current, width, 0);
            embed_row(xt, &m->bigram,
                      blink_bigram_index(previous, current,
                                         m->info.bigram_buckets),
                      width, 1);
            embed_row(xt, positions, t, width, 1);
            const float *segvec = m->segment + (size_t)segment * width;
            for (uint32_t c = 0; c < width; ++c) {
                xt[c] += segvec[c];
            }
        }
        bytes_total += length;
    }
    seg->count = count;
    if (bytes_total == 0u) {
        for (uint32_t i = 0; i < count; ++i) {
            seg->start[i] = seg->length[i] = 0u;
        }
        return 0u;
    }

    /* stem: depthwise local mixing plus one pointwise projection */
    for (uint32_t t = 0; t < bytes_total; ++t) {
        blink_layernorm(s->a + (size_t)t * width, s->x + (size_t)t * width,
                        m->stem.ln_gain, m->stem.ln_bias, width);
    }
    for (uint32_t t0 = 0; t0 < bytes_total; t0 += BLINK_GROUP_ROWS) {
        const uint32_t group = bytes_total - t0 < BLINK_GROUP_ROWS
                                   ? bytes_total - t0 : BLINK_GROUP_ROWS;
        project(s, s->gproj, width, &m->stem.proj,
                             s->a + (size_t)t0 * width, width, group);
        for (uint32_t g = 0; g < group; ++g) {
            float *xt = s->x + (size_t)(t0 + g) * width;
            const float *projected = s->gproj + (size_t)g * width;
            for (uint32_t c = 0; c < width; ++c) {
                xt[c] += projected[c];
            }
        }
    }

    /* Convolve and pool each sequence on its own, then pack the pooled rows.
     * Sequence i starts at byte row o_i and lands at pooled row p_i <= o_i,
     * and p_i + ceil(l_i / stride) <= o_i + l_i = o_{i+1}, so packing in order
     * never overwrites a sequence that has not been pooled yet. */
    uint32_t byte_start = 0u, pooled_total = 0u;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t length = (uint32_t)lengths[i];
        uint32_t pooled = 0u;
        if (length > 0u) {
            float *xs = s->x + (size_t)byte_start * width;
            blink_dwconv_accum(xs, s->a + (size_t)byte_start * width,
                               m->stem.conv, length, width, m->info.conv_width);
            pooled = blink_pool(xs, length, width, stride);
            if (pooled_total != byte_start) {
                memmove(s->x + (size_t)pooled_total * width, xs,
                        (size_t)pooled * width * sizeof(float));
            }
        }
        seg->start[i] = pooled_total;
        seg->length[i] = pooled;
        byte_start += length;
        pooled_total += pooled;
    }

    const uint32_t ffn_width = m->info.ffn_width;
    for (uint32_t b = 0; b < m->info.blocks; ++b) {
        const blink_block_weights *bw = &m->blocks[b];

        /* normalised branch input */
        for (uint32_t t = 0; t < pooled_total; ++t) {
            blink_layernorm(s->a + (size_t)t * width, s->x + (size_t)t * width,
                            bw->ln1_gain, bw->ln1_bias, width);
        }

        /* global term: one mean per sequence, projected back over it */
        for (uint32_t i0 = 0; i0 < count; i0 += BLINK_GROUP_ROWS) {
            const uint32_t group =
                count - i0 < BLINK_GROUP_ROWS ? count - i0 : BLINK_GROUP_ROWS;
            for (uint32_t g = 0; g < group; ++g) {
                const uint32_t first = seg->start[i0 + g];
                const uint32_t length = seg->length[i0 + g];
                float *mean = s->gmean + (size_t)g * width;
                memset(mean, 0, (size_t)width * sizeof(float));
                if (length == 0u) {
                    continue;
                }
                for (uint32_t t = first; t < first + length; ++t) {
                    const float *at = s->a + (size_t)t * width;
                    for (uint32_t c = 0; c < width; ++c) {
                        mean[c] += at[c];
                    }
                }
                const float inverse_length = 1.0f / (float)length;
                for (uint32_t c = 0; c < width; ++c) {
                    mean[c] *= inverse_length;
                }
            }
            project(s, s->gproj, width, &bw->glob, s->gmean, width,
                                 group);
            for (uint32_t g = 0; g < group; ++g) {
                const uint32_t first = seg->start[i0 + g];
                const uint32_t length = seg->length[i0 + g];
                const float *projected = s->gproj + (size_t)g * width;
                for (uint32_t t = first; t < first + length; ++t) {
                    float *xt = s->x + (size_t)t * width;
                    for (uint32_t c = 0; c < width; ++c) {
                        xt[c] += projected[c];
                    }
                }
            }
        }

        /* local term: depthwise convolution accumulated into the residual */
        for (uint32_t i = 0; i < count; ++i) {
            if (seg->length[i] == 0u) {
                continue;
            }
            const size_t offset = (size_t)seg->start[i] * width;
            blink_dwconv_accum(s->x + offset, s->a + offset, bw->conv,
                               seg->length[i], width, m->info.conv_width);
        }

        /* position-wise feed-forward; `a` is free again after the convolution */
        for (uint32_t t = 0; t < pooled_total; ++t) {
            blink_layernorm(s->a + (size_t)t * width, s->x + (size_t)t * width,
                            bw->ln2_gain, bw->ln2_bias, width);
        }
        for (uint32_t t0 = 0; t0 < pooled_total; t0 += BLINK_GROUP_ROWS) {
            const uint32_t group = pooled_total - t0 < BLINK_GROUP_ROWS
                                       ? pooled_total - t0 : BLINK_GROUP_ROWS;
            project(s, s->ffn, ffn_width, &bw->fc1,
                                 s->a + (size_t)t0 * width, width, group);
            for (uint32_t f = 0; f < group * ffn_width; ++f) {
                s->ffn[f] = relu2(s->ffn[f]);
            }
            project(s, s->gproj, width, &bw->fc2, s->ffn, ffn_width,
                                 group);
            for (uint32_t g = 0; g < group; ++g) {
                float *xt = s->x + (size_t)(t0 + g) * width;
                const float *projected = s->gproj + (size_t)g * width;
                for (uint32_t c = 0; c < width; ++c) {
                    xt[c] += projected[c];
                }
            }
        }

        if (bw->mixer.present) {
            mix_positions(s, &bw->mixer, seg, pooled_total);
        }
    }

    /* Only the question reads the state, and only after its own blocks have
     * run. The state cannot (it is encoded first, and letting it look at the
     * question would destroy the cache); the options do not, so a menu stays
     * independent of both. */
    if (m->cross.present && segment == BLINK_SEG_QUESTION &&
        s->state_positions > 0u) {
        cross_attend(s, pooled_total, s->state_positions);
    }
    return pooled_total;
}

/* Project encoded tokens into the key/value cache starting at `offset`.
 *
 * Each head's slice of a key is stored at unit length. The head only ever
 * uses keys through a cosine, so normalising here, once per cached position,
 * rather than once per option per head is the same computation done fewer
 * times -- and it keeps a cached state bit-identical to a re-encoded one. */
static void write_kv(blink_session *s, uint32_t length, uint32_t offset)
{
    const blink_model *m = s->model;
    const uint32_t width = m->info.width;
    const uint32_t rank = m->info.rank;
    const uint32_t head_dim = rank / m->info.heads;
    for (uint32_t t = 0; t < length; ++t) {
        blink_layernorm(s->a + (size_t)t * width, s->x + (size_t)t * width,
                        m->ctx_ln_gain, m->ctx_ln_bias, width);
    }
    project(s, s->keys + (size_t)offset * rank, rank, &m->wk, s->a,
                         width, length);
    project(s, s->values + (size_t)offset * rank, rank, &m->wv, s->a,
                         width, length);
    for (uint32_t t = 0; t < length; ++t) {
        float *key = s->keys + (size_t)(offset + t) * rank;
        for (uint32_t h = 0; h < m->info.heads; ++h) {
            blink_l2_normalize(key + (size_t)h * head_dim, head_dim);
        }
    }
}

/* ------------------------------------------------------------ public API */

blink_status blink_state_set(blink_session *s, const char *state, size_t length)
{
    if (!s || (!state && length)) {
        return BLINK_E_INVALID;
    }
    if (length > s->limits.max_state) {
        return BLINK_E_LIMIT;
    }
    s->has_state = 0;
    s->has_score = 0;
    s->state_positions = 0u;   /* so the state cannot cross-attend to itself */
    s->state_bytes = (uint32_t)length;
    blink_segments seg;
    s->state_positions =
        encode_sequences(s, &state, &length, 1u, BLINK_SEG_STATE,
                         &s->model->pos_state, &seg);
    write_kv(s, s->state_positions, 0u);
    if (s->model->cross.present) {
        write_cross_kv(s, s->state_positions);
    }
    s->has_state = 1;
    return BLINK_OK;
}

blink_status blink_menu_set(blink_session *s, const char *const *options,
                            const size_t *lengths, uint32_t count)
{
    if (!s || !options || !lengths) {
        return BLINK_E_INVALID;
    }
    if (count < 2u || count > s->limits.max_options) {
        return BLINK_E_LIMIT;
    }
    const blink_model *m = s->model;
    const uint32_t width = m->info.width;

    for (uint32_t n = 0; n < count; ++n) {
        if (!options[n] || lengths[n] == 0u) {
            return BLINK_E_INVALID;
        }
        if (lengths[n] > s->limits.max_option) {
            return BLINK_E_LIMIT;
        }
    }
    /* Options are encoded as many at a time as the encoder scratch holds. */
    blink_segments seg;
    for (uint32_t first = 0; first < count; first += seg.count) {
        const uint32_t take = segments_that_fit(s, lengths + first,
                                                count - first, UINT32_MAX);
        encode_sequences(s, options + first, lengths + first, take,
                         BLINK_SEG_OPTION, &m->pos_option, &seg);
        for (uint32_t i = 0; i < take; ++i) {
            float *vector = s->optvec + (size_t)(first + i) * width;
            memset(vector, 0, (size_t)width * sizeof(float));
            for (uint32_t t = seg.start[i]; t < seg.start[i] + seg.length[i];
                 ++t) {
                const float *xt = s->x + (size_t)t * width;
                for (uint32_t c = 0; c < width; ++c) {
                    vector[c] += xt[c];
                }
            }
            const float inverse = 1.0f / (float)seg.length[i];
            for (uint32_t c = 0; c < width; ++c) {
                vector[c] *= inverse;
            }
        }
    }
    s->menu_size = count;
    s->has_menu = 1;
    s->has_score = 0;
    return BLINK_OK;
}

/* Project the cached option vectors into attention queries, modulating them
 * with a summary of the question when the model carries FiLM weights.
 *
 * `summary` is the mean of the question's pooled hidden states, or NULL when
 * the question is empty; an empty question leaves the option vectors alone,
 * because gamma and beta are then both zero. */
static void build_queries(blink_session *s, const float *summary)
{
    const blink_model *m = s->model;
    const uint32_t width = m->info.width;
    const uint32_t rank = m->info.rank;

    if (m->info.film && summary) {
        blink_layernorm(s->tmpw, summary, m->film_ln_gain, m->film_ln_bias,
                        width);
        blink_matmul_f32(s->gmean, m->film_gamma, s->tmpw, width, width);
        blink_matmul_f32(s->gproj, m->film_beta, s->tmpw, width, width);
    } else {
        memset(s->gmean, 0, (size_t)width * sizeof(float));
        memset(s->gproj, 0, (size_t)width * sizeof(float));
    }

    for (uint32_t n = 0; n < s->menu_size; ++n) {
        const float *option = s->optvec + (size_t)n * width;
        for (uint32_t c = 0; c < width; ++c) {
            s->tmpw[c] = option[c] * (1.0f + s->gmean[c]) + s->gproj[c];
        }
        blink_layernorm(s->a + (size_t)n % BLINK_GROUP_ROWS * width, s->tmpw,
                        m->opt_ln_gain, m->opt_ln_bias, width);
        if (n % BLINK_GROUP_ROWS == BLINK_GROUP_ROWS - 1u ||
            n + 1u == s->menu_size) {
            const uint32_t first = n - n % BLINK_GROUP_ROWS;
            project(s, s->queries + (size_t)first * rank, rank,
                                 &m->wq, s->a, width, n - first + 1u);
        }
    }
}

/* The option head for one question whose keys and values sit at cached
 * positions `first` .. `first + positions - 1`, after the state's. The
 * context it attends over is the state followed by that question, exactly
 * as if the question had been the only one encoded. */
static void score_head(blink_session *s, uint32_t first, uint32_t positions,
                       float *probabilities)
{
    const blink_model *m = s->model;
    const uint32_t rank = m->info.rank;
    const uint32_t heads = m->info.heads;
    const uint32_t head_dim = rank / heads;
    const uint32_t options = s->menu_size;
    const uint32_t state = s->state_positions;
    const uint32_t context = state + positions;

    /* Queries, keys and the attended vector are unit length, so a score is a
     * cosine and lives in [-1, 1]. Multiplying by sqrt(head_dim) restores
     * enough range for the softmax to become sharp while keeping the scores
     * bounded whatever the projections weigh. The logit is then a cosine too,
     * bounded in [-heads, heads] before logit_scale. */
    const float span = sqrtf((float)head_dim);

    for (uint32_t n = 0; n < options; ++n) {
        float *attention = s->attn + (size_t)n * s->context_capacity;
        memset(attention, 0, (size_t)context * sizeof(float));
        float logit = 0.0f;

        for (uint32_t h = 0; h < heads; ++h) {
            const uint32_t base = h * head_dim;
            const float *source = s->queries + (size_t)n * rank + base;
            memcpy(s->qhead, source, (size_t)head_dim * sizeof(float));
            blink_l2_normalize(s->qhead, head_dim);

            /* The context is the state's cached rows, then this question's,
             * which start at row `first` and need not follow the state's
             * directly in a batch: two contiguous runs. */
            blink_attention_scores(s->scores, s->keys + base, rank, s->qhead,
                                   head_dim, state, span);
            blink_attention_scores(s->scores + state,
                                   s->keys + (size_t)first * rank + base, rank,
                                   s->qhead, head_dim, positions, span);
            blink_softmax(s->scores, context);

            memset(s->headbuf, 0, (size_t)head_dim * sizeof(float));
            blink_attention_accum(s->headbuf, s->values + base, rank,
                                  s->scores, head_dim, state);
            blink_attention_accum(s->headbuf,
                                  s->values + (size_t)first * rank + base, rank,
                                  s->scores + state, head_dim, positions);
            for (uint32_t l = 0; l < context; ++l) {
                attention[l] += s->scores[l];
            }
            blink_l2_normalize(s->headbuf, head_dim);
            logit += blink_dot(s->qhead, s->headbuf, head_dim);
        }

        for (uint32_t l = 0; l < context; ++l) {
            attention[l] /= (float)heads;
        }
        s->logits[n] = logit * m->logit_scale;
    }

    const float inverse_temperature = 1.0f / m->info.temperature;
    for (uint32_t n = 0; n < options; ++n) {
        probabilities[n] = s->logits[n] * inverse_temperature;
    }
    blink_softmax(probabilities, options);
}

static void fill_result(const blink_session *s, const float *probabilities,
                        blink_result *result, double encode_seconds,
                        double head_seconds)
{
    const uint32_t options = s->menu_size;
    uint32_t argmax = 0u;
    float second = -1.0f;
    for (uint32_t n = 1; n < options; ++n) {
        if (probabilities[n] > probabilities[argmax]) {
            argmax = n;
        }
    }
    for (uint32_t n = 0; n < options; ++n) {
        if (n != argmax && probabilities[n] > second) {
            second = probabilities[n];
        }
    }
    float entropy = 0.0f;
    for (uint32_t n = 0; n < options; ++n) {
        if (probabilities[n] > 0.0f) {
            entropy -= probabilities[n] * logf(probabilities[n]);
        }
    }
    result->options = options;
    result->argmax = argmax;
    result->confidence = probabilities[argmax];
    result->entropy = entropy;
    result->margin = probabilities[argmax] - second;
    result->state_bytes = s->state_bytes;
    result->question_bytes = s->question_bytes;
    result->context_positions = s->state_positions + s->question_positions;
    result->encode_seconds = encode_seconds;
    result->head_seconds = head_seconds;
}

blink_status blink_score_batch(blink_session *s, const char *const *questions,
                               const size_t *lengths, uint32_t count,
                               float *probabilities, blink_result *results)
{
    if (!s || !questions || !lengths || !probabilities || count == 0u) {
        return BLINK_E_INVALID;
    }
    if (!s->has_state || !s->has_menu) {
        return BLINK_E_STATE;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (!questions[i] && lengths[i]) {
            return BLINK_E_INVALID;
        }
        if (lengths[i] > s->limits.max_question) {
            return BLINK_E_LIMIT;
        }
        /* An empty question against an empty state leaves nothing to attend
         * to. Refused before any work, so a failed batch changes nothing. */
        if (lengths[i] == 0u && s->state_positions == 0u) {
            return BLINK_E_INVALID;
        }
    }

    const blink_model *m = s->model;
    const uint32_t width = m->info.width;
    const uint32_t options = s->menu_size;
    const uint32_t state = s->state_positions;
    /* Question keys and values go after the state's, so a pass may hold as
     * many pooled question positions as the cache has left. */
    const uint32_t room = s->context_capacity - state;

    blink_segments seg;
    for (uint32_t first = 0; first < count; first += seg.count) {
        const uint32_t take =
            segments_that_fit(s, lengths + first, count - first, room);
        const double encode_start = monotonic_seconds();
        encode_sequences(s, questions + first, lengths + first, take,
                         BLINK_SEG_QUESTION, &m->pos_quest, &seg);
        uint32_t pooled = 0u;
        for (uint32_t i = 0; i < take; ++i) {
            pooled += seg.length[i];
        }
        write_kv(s, pooled, state);
        const double shared = (monotonic_seconds() - encode_start) / take;

        for (uint32_t i = 0; i < take; ++i) {
            const uint32_t q = first + i;
            const double query_start = monotonic_seconds();
            s->question_bytes = (uint32_t)lengths[q];
            s->question_positions = seg.length[i];

            /* Summarise the question, then rebuild the queries. */
            const float *summary = NULL;
            if (seg.length[i]) {
                memset(s->qsum, 0, (size_t)width * sizeof(float));
                for (uint32_t t = seg.start[i];
                     t < seg.start[i] + seg.length[i]; ++t) {
                    const float *xt = s->x + (size_t)t * width;
                    for (uint32_t c = 0; c < width; ++c) {
                        s->qsum[c] += xt[c];
                    }
                }
                const float inverse_positions = 1.0f / (float)seg.length[i];
                for (uint32_t c = 0; c < width; ++c) {
                    s->qsum[c] *= inverse_positions;
                }
                summary = s->qsum;
            }
            build_queries(s, summary);
            const double head_start = monotonic_seconds();

            float *out = probabilities + (size_t)q * options;
            score_head(s, state + seg.start[i], seg.length[i], out);
            s->has_score = 1;
            if (results) {
                fill_result(s, out, &results[q],
                            shared + (head_start - query_start),
                            monotonic_seconds() - head_start);
            }
        }
    }
    return BLINK_OK;
}

blink_status blink_score(blink_session *s, const char *question, size_t length,
                         float *probabilities, blink_result *result)
{
    if (!s || !probabilities || (!question && length)) {
        return BLINK_E_INVALID;
    }
    return blink_score_batch(s, &question, &length, 1u, probabilities, result);
}

blink_status blink_decide(blink_session *s, const char *state,
                          size_t state_length, const char *question,
                          size_t question_length, const char *const *options,
                          const size_t *lengths, uint32_t count,
                          float *probabilities, blink_result *result)
{
    blink_status status = blink_state_set(s, state, state_length);
    if (status != BLINK_OK) return status;
    status = blink_menu_set(s, options, lengths, count);
    if (status != BLINK_OK) return status;
    return blink_score(s, question, question_length, probabilities, result);
}

const float *blink_last_logits(const blink_session *s, uint32_t *count)
{
    if (!s || !s->has_score) {
        return NULL;
    }
    if (count) {
        *count = s->menu_size;
    }
    return s->logits;
}

uint32_t blink_last_attention(const blink_session *s, uint32_t option,
                              float *out, uint32_t capacity)
{
    if (!s || !s->has_score || !out || option >= s->menu_size) {
        return 0u;
    }
    uint32_t context = s->state_positions + s->question_positions;
    if (context > capacity) {
        context = capacity;
    }
    memcpy(out, s->attn + (size_t)option * s->context_capacity,
           (size_t)context * sizeof(float));
    return context;
}
