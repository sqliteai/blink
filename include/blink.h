/* blink.h -- embeddable one-pass typed-decision runtime.
 *
 * Blink scores a runtime-defined list of text options against a state and a
 * question in a single forward pass. There is no token generation, no decoding
 * loop and no dynamic allocation on the scoring path.
 *
 * C99. Depends only on the C standard library and libm.
 *
 * Threading: every object is single-threaded. A blink_model is immutable once
 * opened and may be shared by any number of sessions on any number of threads.
 * A blink_session must be used by one thread at a time.
 *
 * Memory: the caller owns all memory. blink_model_open_file maps the weight
 * file read-only and never copies it; blink_session_init places an entire
 * session inside a caller-supplied buffer.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef BLINK_H
#define BLINK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Library version, semantic versioning. BLINK_VERSION_STRING is built from
 * the three numbers, so they cannot disagree, and blink_version() returns the
 * same string from the compiled library: compare the two to catch a header
 * and a library from different releases. BLINK_VERSION_NUMBER orders
 * releases for the preprocessor, as MAJOR * 10000 + MINOR * 100 + PATCH:
 *
 *     #if BLINK_VERSION_NUMBER >= 100   (0.1.0 or later)
 */
#define BLINK_VERSION_MAJOR 0
#define BLINK_VERSION_MINOR 1
#define BLINK_VERSION_PATCH 0

#define BLINK_STRINGIFY_(x) #x
#define BLINK_STRINGIFY(x) BLINK_STRINGIFY_(x)
#define BLINK_VERSION_STRING                                                   \
    BLINK_STRINGIFY(BLINK_VERSION_MAJOR) "."                                   \
    BLINK_STRINGIFY(BLINK_VERSION_MINOR) "."                                   \
    BLINK_STRINGIFY(BLINK_VERSION_PATCH)
#define BLINK_VERSION_NUMBER                                                   \
    (BLINK_VERSION_MAJOR * 10000 + BLINK_VERSION_MINOR * 100 + BLINK_VERSION_PATCH)

/* The binary interface: bumped when a struct layout or a function signature
 * changes incompatibly, independently of the release number. */
#define BLINK_ABI_VERSION 1u

/* Hard ceilings enforced by the runtime regardless of the model file. */
#define BLINK_MAX_OPTIONS 64u
#define BLINK_MAX_SEQ 4096u

typedef enum blink_status {
    BLINK_OK = 0,
    BLINK_E_INVALID,     /* caller passed a null or out-of-range argument   */
    BLINK_E_IO,          /* the weight file could not be read or mapped     */
    BLINK_E_FORMAT,      /* the weight file is not a valid .blink container */
    BLINK_E_VERSION,     /* the container version is not supported          */
    BLINK_E_CHECKSUM,    /* the container failed its CRC-32 check           */
    BLINK_E_NOSPACE,     /* the supplied arena is too small                 */
    BLINK_E_LIMIT,       /* input exceeds the session limits                */
    BLINK_E_STATE        /* required step (state / menu) not performed yet  */
} blink_status;

const char *blink_status_string(blink_status status);

/* Version of the compiled library, as "MAJOR.MINOR.PATCH": the
 * BLINK_VERSION_STRING it was built with. */
const char *blink_version(void);

/* The numeric backend this library is running: "scalar", "neon", "sse2" or
 * "avx2" (x86-64, chosen at run time, BLINK_X86_SIMD to force), "accelerate"
 * (make ACCELERATE=1), or "w8a8-scalar", "w8a8-sdot", "w8a8-i8mm",
 * "w8a8-sse2" or "w8a8-avx2" (make W8A8=1, chosen for the CPU at run time:
 * SDOT on Apple silicon, I8MM on other Arm cores that have it, AVX2 on x86-64
 * when present; BLINK_W8A8_KERNEL to force). The default builds -- scalar,
 * neon, sse2, avx2 -- compute the normative W8A32 product; see
 * docs/ARCHITECTURE.md. */
const char *blink_backend(void);

/* ------------------------------------------------------------------ model */

typedef struct blink_model blink_model;

/* Immutable description of a loaded model. */
typedef struct blink_model_info {
    uint32_t width;          /* residual width                              */
    uint32_t blocks;         /* encoder blocks                              */
    uint32_t ffn_width;      /* inner width of the block feed-forward       */
    uint32_t rank;           /* total attention rank (heads * head_dim)     */
    uint32_t heads;          /* attention heads in the option head          */
    uint32_t conv_width;     /* depthwise convolution kernel size           */
    uint32_t stride;         /* byte positions merged after the stem        */
    uint32_t mixer_blocks;   /* trailing blocks carrying self-attention     */
    uint32_t film;           /* 1 when the question conditions the queries  */
    uint32_t cross;          /* 1 when the question attends to the state    */
    uint32_t bigram_buckets; /* hashed byte-bigram table rows               */
    uint32_t max_state;      /* bytes of state the model was trained for    */
    uint32_t max_question;   /* bytes of question                           */
    uint32_t max_option;     /* bytes per option                            */
    float temperature;       /* calibration temperature applied to logits   */
    uint64_t parameters;     /* total scalar parameters                     */
    uint64_t weights_bytes;  /* bytes of weight data (mapped, not copied)   */
    char name[32];           /* preset name, NUL-terminated                 */
} blink_model_info;

/* Open a .blink container. The file is mapped read-only and is not copied.
 * Pass verify_checksum = 0 to skip the CRC-32 pass on very large files. */
blink_model *blink_model_open_file(const char *path, int verify_checksum,
                                   blink_status *status);

/* Open a container already resident in memory. The buffer is borrowed: it must
 * outlive the model and must stay unmodified. Useful for weights linked into
 * the binary with a .incbin or an xxd dump. */
blink_model *blink_model_open_memory(const void *data, size_t size,
                                     int verify_checksum, blink_status *status);

void blink_model_close(blink_model *model);
void blink_model_get_info(const blink_model *model, blink_model_info *out);

/* ---------------------------------------------------------------- session */

typedef struct blink_session blink_session;

/* Per-session ceilings. Any field left at 0 takes the model's own value.
 * Smaller limits mean a smaller arena. */
typedef struct blink_limits {
    uint32_t max_state;    /* bytes of state accepted by blink_state_set    */
    uint32_t max_question; /* bytes of question accepted by blink_score     */
    uint32_t max_options;  /* options accepted by blink_menu_set            */
    uint32_t max_option;   /* bytes per option                              */
} blink_limits;

/* Bytes required for a session with these limits. Deterministic: the same
 * model and limits always give the same size, and no allocation ever happens
 * outside this block. */
size_t blink_session_size(const blink_model *model, const blink_limits *limits);

/* Place a session in caller memory. `memory` must be at least
 * blink_session_size bytes and aligned to at least 16 bytes (malloc,
 * a static array or a stack buffer with alignas all qualify). The session
 * borrows the model; the model must outlive it. */
blink_session *blink_session_init(void *memory, size_t size,
                                  const blink_model *model,
                                  const blink_limits *limits,
                                  blink_status *status);

/* Convenience wrapper that mallocs its own arena. blink_session_free releases
 * it. Sessions created with blink_session_init must not be passed here. */
blink_session *blink_session_create(const blink_model *model,
                                    const blink_limits *limits,
                                    blink_status *status);
void blink_session_free(blink_session *session);

/* -------------------------------------------------------------- inference */

/* Encode the state once. The result is cached in the session and reused by
 * every later blink_score call until the state is replaced. Encoding a state
 * is the expensive part of a decision; questions and menus are cheap. */
blink_status blink_state_set(blink_session *session, const char *state,
                             size_t length);

/* Encode a menu once. Options are runtime-defined: the count and the text may
 * change on every call. The encoded option vectors are cached until replaced.
 *
 * The cheap projection from an option vector to an attention query happens in
 * blink_score, because the query is conditioned on the question. Encoding the
 * option text -- the expensive part -- still happens only here. */
blink_status blink_menu_set(blink_session *session, const char *const *options,
                            const size_t *lengths, uint32_t count);

/* Per-decision timings and diagnostics. All times are in seconds. */
typedef struct blink_result {
    uint32_t options;       /* number of probabilities written              */
    uint32_t argmax;        /* index of the highest probability             */
    float confidence;       /* probability of argmax, after temperature     */
    float entropy;          /* natural-log entropy of the distribution      */
    float margin;           /* top-1 probability minus top-2 probability    */
    uint32_t state_bytes;   /* state bytes accepted                         */
    uint32_t question_bytes;
    uint32_t context_positions; /* pooled positions the head attended over   */
    double encode_seconds;  /* question encoding only                       */
    double head_seconds;    /* option attention and softmax                 */
} blink_result;

/* Score the cached menu against the cached state under `question`.
 * `probabilities` receives one value per option and must have room for the
 * menu size. `result` may be NULL. Requires blink_state_set and
 * blink_menu_set to have succeeded at least once.
 *
 * This call performs no allocation and no I/O. */
blink_status blink_score(blink_session *session, const char *question,
                         size_t length, float *probabilities,
                         blink_result *result);

/* Score the cached menu under `count` questions in one call. Question i's
 * probabilities go to probabilities[i * menu_size .. (i + 1) * menu_size - 1],
 * and its diagnostics to results[i] when `results` is not NULL.
 *
 * The output is bit-identical to calling blink_score once per question, in
 * order. It is faster because the questions are encoded side by side, so the
 * weight projections run over several questions' positions at once; the
 * session arena is the same size, and the batch is split internally into
 * passes that fit it. encode_seconds shares each pass's encoding time evenly
 * between its questions. Every question is validated before any is scored, so
 * a batch that fails changes nothing. Afterwards blink_last_logits and
 * blink_last_attention describe the last question. */
blink_status blink_score_batch(blink_session *session,
                               const char *const *questions,
                               const size_t *lengths, uint32_t count,
                               float *probabilities, blink_result *results);

/* One-shot form: sets state, sets menu, scores. Convenient, but it re-encodes
 * the state on every call. Prefer the three-step form in a loop. */
blink_status blink_decide(blink_session *session, const char *state,
                          size_t state_length, const char *question,
                          size_t question_length, const char *const *options,
                          const size_t *lengths, uint32_t count,
                          float *probabilities, blink_result *result);

/* Raw pre-temperature logits of the last blink_score, for calibration work.
 * Returns NULL before the first successful score. */
const float *blink_last_logits(const blink_session *session, uint32_t *count);

/* Attention distribution over context positions for one option of the last
 * score, averaged over heads. There is one value per pooled position, not per
 * byte: see blink_result.context_positions. Writes at most `capacity` values
 * and returns how many were written. Used by the audit tooling to show
 * that a decision depends on the state. */
uint32_t blink_last_attention(const blink_session *session, uint32_t option,
                              float *out, uint32_t capacity);

#ifdef __cplusplus
}
#endif

#endif /* BLINK_H */
