/* Numeric kernels. Every loop here is the normative definition of the Blink
 * forward pass: the Python reference implementation in
 * python/blink_train/reference.py mirrors these loops term by term, and
 * tests/parity checks that the two agree.
 *
 * The loops are written so that a vectorizing compiler can widen them without
 * reordering the accumulation across more than one lane group; the scalar form
 * is the reference and the build pins -ffp-contract=off so a fused multiply-add
 * cannot silently change results between platforms.
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "blink_internal.h"

/* An explicit SIMD path for the one kernel that dominates the profile.
 *
 * BLINK_SCALAR_ONLY=1 disables it. The scalar loop below stays the normative
 * definition: it is what python/blink_train/reference.py mirrors, and
 * tests/c/test_kernels.c checks that the two paths agree. They are not
 * bit-identical, because widening the loop changes the order of the fp32
 * additions; the difference is bounded by fp32 rounding and is well inside the
 * parity tolerance. */
#if !defined(BLINK_SCALAR_ONLY) && defined(__ARM_NEON)
#include <arm_neon.h>
#define BLINK_SIMD_NEON 1
#endif

/* The x86-64 equivalent. SSE2 is part of x86-64 itself, so that kernel always
 * exists; the AVX2 + FMA kernel is compiled with a function-level target and
 * chosen at run time when the CPU reports both, so one binary runs on any
 * x86-64 machine. BLINK_X86_SIMD=scalar|sse2|avx2 overrides the choice. */
#if !defined(BLINK_SCALAR_ONLY) && defined(__x86_64__) &&                      \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define BLINK_SIMD_X86 1
#define BLINK_AVX2_TARGET __attribute__((target("avx2,fma")))
#endif

/* Optional macOS backend: projections through Accelerate's BLAS, which on
 * Apple silicon runs on the CPU's matrix unit rather than on NEON.
 *
 * Built only with `make ACCELERATE=1`. It needs the dense fp32 copy of each
 * projection that blink_model.c makes at open time; a tensor without one (the
 * test fixtures build their own) falls back to the kernels below. The result
 * is the same product summed in a different order, with the row scale folded
 * into the weights, so it agrees with the scalar definition to fp32 rounding
 * but not bit for bit -- and a row's result may depend on how many rows were
 * multiplied with it. Everything in this file still references no allocator,
 * but Accelerate itself allocates a buffer inside every product large enough
 * to use the matrix unit (16 per blink-tiny decision, measured by interposing
 * malloc), so this build does not keep the zero-allocation guarantee. */
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#endif

/* Optional W8A8 backend (`make W8A8=1`): activations quantized to int8 per
 * row and multiplied with integer instructions. SDOT is used when the
 * compiler targets it (every Apple silicon core does); I8MM, which the
 * default arm64 target does not enable, is compiled with a function-level
 * target and chosen at run time when the CPU reports it. */
#if defined(BLINK_W8A8) && BLINK_W8A8
#if defined(BLINK_SIMD_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define BLINK_HAVE_SDOT 1
#endif
#if defined(BLINK_SIMD_NEON) && defined(__aarch64__) &&                        \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 10))
#define BLINK_HAVE_I8MM 1
#define BLINK_I8MM_TARGET __attribute__((target("arch=armv8.2-a+dotprod+i8mm")))
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#endif
#endif
#endif

float blink_dot(const float *restrict a, const float *restrict b, uint32_t n)
{
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
    /* The strict-order loop below cannot be vectorized under
     * -ffp-contract=off; vDSP sums in its own order. */
    float result = 0.0f;
    vDSP_dotpr(a, 1, b, 1, &result, n);
    return result;
#else
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

void blink_layernorm(float *restrict out, const float *restrict in,
                     const float *restrict gain, const float *restrict bias,
                     uint32_t width)
{
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
    /* The same three passes through vDSP: centre, mean square, then scale,
     * gain and bias. `out` holds the centred values in between. */
    float mean = 0.0f, variance = 0.0f;
    vDSP_meanv(in, 1, &mean, width);
    const float negated = -mean;
    vDSP_vsadd(in, 1, &negated, out, 1, width);
    vDSP_measqv(out, 1, &variance, width);
    const float inverse = 1.0f / sqrtf(variance + 1e-5f);
    vDSP_vsmul(out, 1, &inverse, out, 1, width);
    vDSP_vma(out, 1, gain, 1, bias, 1, out, 1, width);
#else
    float mean = 0.0f;
    for (uint32_t i = 0; i < width; ++i) {
        mean += in[i];
    }
    mean /= (float)width;

    float variance = 0.0f;
    for (uint32_t i = 0; i < width; ++i) {
        const float centred = in[i] - mean;
        variance += centred * centred;
    }
    variance /= (float)width;

    const float inv = 1.0f / sqrtf(variance + 1e-5f);
    for (uint32_t i = 0; i < width; ++i) {
        out[i] = (in[i] - mean) * inv * gain[i] + bias[i];
    }
#endif
}

/* The normative definition. Every other implementation of this product, in C
 * or in Python, must agree with this loop. */
static float dot_i8_scalar(const int8_t *restrict row, const float *restrict x,
                           uint32_t cols)
{
    float sum = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
        sum += (float)row[c] * x[c];
    }
    return sum;
}

#ifdef BLINK_SIMD_NEON
/* Sixteen int8 weights per iteration: widen to int16, then to int32, convert
 * to fp32 and accumulate into four independent lanes. The conversion is the
 * expensive part of a W8A32 product, so doing it sixteen at a time is where
 * the speedup comes from rather than from the multiply itself. */
static float dot_i8_neon(const int8_t *restrict row, const float *restrict x,
                         uint32_t cols)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    uint32_t c = 0;
    for (; c + 16u <= cols; c += 16u) {
        const int8x16_t bytes = vld1q_s8(row + c);
        const int16x8_t low = vmovl_s8(vget_low_s8(bytes));
        const int16x8_t high = vmovl_s8(vget_high_s8(bytes));
        acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))),
                         vld1q_f32(x + c));
        acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
                         vld1q_f32(x + c + 4u));
        acc2 = vfmaq_f32(acc2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))),
                         vld1q_f32(x + c + 8u));
        acc3 = vfmaq_f32(acc3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))),
                         vld1q_f32(x + c + 12u));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1),
                                     vaddq_f32(acc2, acc3)));
    for (; c < cols; ++c) {
        sum += (float)row[c] * x[c];
    }
    return sum;
}
#endif

#ifdef BLINK_SIMD_X86
/* 0 scalar, 1 SSE2, 2 AVX2 + FMA. Decided on first use and then fixed, so a
 * process never mixes two paths: the row-blocked and one-vector products
 * below are bit-identical only within one path. */
static int x86_level = -1;

static int x86_detect(void)
{
    int level = 1;
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        level = 2;
    }
    const char *requested = getenv("BLINK_X86_SIMD");
    if (requested) {
        if (strcmp(requested, "scalar") == 0) {
            level = 0;
        } else if (strcmp(requested, "sse2") == 0) {
            level = 1;
        }
    }
    return level;
}

int blink_x86_level(void)
{
    if (x86_level < 0) {
        x86_level = x86_detect();
    }
    return x86_level;
}

/* For tests/c/test_kernels.c: asking for AVX2 on a CPU without it is
 * ignored. Returns the level now in use. */
int blink_x86_force_level(int level)
{
    if (level == 0 || level == 1) {
        x86_level = level;
    } else if (level == 2 && __builtin_cpu_supports("avx2") &&
               __builtin_cpu_supports("fma")) {
        x86_level = 2;
    }
    return blink_x86_level();
}

static inline __attribute__((always_inline)) float hsum_sse(__m128 v)
{
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
    return _mm_cvtss_f32(v);
}

/* Sixteen int8 weights per step, sign-extended with SSE2 alone (unpack a byte
 * onto itself, then an arithmetic shift) to four vectors of four floats,
 * accumulated into two lanes groups per input vector. `n` input vectors share
 * each widened weight row; it is a compile-time constant at every call. */
static inline __attribute__((always_inline)) void
rows_i8_sse2(float *restrict out, size_t out_stride, const int8_t *restrict q,
             const float *restrict scale, uint32_t rows, uint32_t cols,
             const float *restrict x, size_t x_stride, uint32_t n)
{
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *restrict row = q + (size_t)r * cols;
        __m128 acc[BLINK_ROW_BLOCK][2];
        for (uint32_t v = 0; v < n; ++v) {
            acc[v][0] = _mm_setzero_ps();
            acc[v][1] = _mm_setzero_ps();
        }
        uint32_t c = 0;
        for (; c + 16u <= cols; c += 16u) {
            const __m128i bytes = _mm_loadu_si128((const __m128i *)(row + c));
            const __m128i lo16 = _mm_srai_epi16(_mm_unpacklo_epi8(bytes, bytes), 8);
            const __m128i hi16 = _mm_srai_epi16(_mm_unpackhi_epi8(bytes, bytes), 8);
            const __m128 w0 = _mm_cvtepi32_ps(
                _mm_srai_epi32(_mm_unpacklo_epi16(lo16, lo16), 16));
            const __m128 w1 = _mm_cvtepi32_ps(
                _mm_srai_epi32(_mm_unpackhi_epi16(lo16, lo16), 16));
            const __m128 w2 = _mm_cvtepi32_ps(
                _mm_srai_epi32(_mm_unpacklo_epi16(hi16, hi16), 16));
            const __m128 w3 = _mm_cvtepi32_ps(
                _mm_srai_epi32(_mm_unpackhi_epi16(hi16, hi16), 16));
            for (uint32_t v = 0; v < n; ++v) {
                const float *restrict xv = x + v * x_stride + c;
                acc[v][0] = _mm_add_ps(acc[v][0], _mm_mul_ps(w0, _mm_loadu_ps(xv)));
                acc[v][1] = _mm_add_ps(acc[v][1],
                                       _mm_mul_ps(w1, _mm_loadu_ps(xv + 4u)));
                acc[v][0] = _mm_add_ps(acc[v][0],
                                       _mm_mul_ps(w2, _mm_loadu_ps(xv + 8u)));
                acc[v][1] = _mm_add_ps(acc[v][1],
                                       _mm_mul_ps(w3, _mm_loadu_ps(xv + 12u)));
            }
        }
        for (uint32_t v = 0; v < n; ++v) {
            const float *restrict xv = x + v * x_stride;
            float sum = hsum_sse(_mm_add_ps(acc[v][0], acc[v][1]));
            for (uint32_t t = c; t < cols; ++t) {
                sum += (float)row[t] * xv[t];
            }
            out[v * out_stride + r] = sum * scale[r];
        }
    }
}

BLINK_AVX2_TARGET
static inline __attribute__((always_inline)) float hsum_avx(__m256 v)
{
    return hsum_sse(_mm_add_ps(_mm256_castps256_ps128(v),
                               _mm256_extractf128_ps(v, 1)));
}

/* The same product with AVX2: eight weights widened per instruction
 * (vpmovsxbd) and fused multiply-adds into two 8-lane accumulators per input
 * vector. */
BLINK_AVX2_TARGET
static inline __attribute__((always_inline)) void
rows_i8_avx2_n(float *restrict out, size_t out_stride, const int8_t *restrict q,
               const float *restrict scale, uint32_t rows, uint32_t cols,
               const float *restrict x, size_t x_stride, uint32_t n)
{
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *restrict row = q + (size_t)r * cols;
        __m256 acc[BLINK_ROW_BLOCK][2];
        for (uint32_t v = 0; v < n; ++v) {
            acc[v][0] = _mm256_setzero_ps();
            acc[v][1] = _mm256_setzero_ps();
        }
        uint32_t c = 0;
        for (; c + 16u <= cols; c += 16u) {
            const __m128i bytes = _mm_loadu_si128((const __m128i *)(row + c));
            const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            const __m256 w1 =
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bytes, 8)));
            for (uint32_t v = 0; v < n; ++v) {
                const float *restrict xv = x + v * x_stride + c;
                acc[v][0] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xv), acc[v][0]);
                acc[v][1] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xv + 8u), acc[v][1]);
            }
        }
        for (uint32_t v = 0; v < n; ++v) {
            const float *restrict xv = x + v * x_stride;
            float sum = hsum_avx(_mm256_add_ps(acc[v][0], acc[v][1]));
            for (uint32_t t = c; t < cols; ++t) {
                sum += (float)row[t] * xv[t];
            }
            out[v * out_stride + r] = sum * scale[r];
        }
    }
}

BLINK_AVX2_TARGET
static void rows_i8_avx2(float *restrict out, size_t out_stride,
                         const int8_t *restrict q, const float *restrict scale,
                         uint32_t rows, uint32_t cols, const float *restrict x,
                         size_t x_stride, uint32_t n)
{
    switch (n) {
    case 4u: rows_i8_avx2_n(out, out_stride, q, scale, rows, cols, x, x_stride, 4u); break;
    case 3u: rows_i8_avx2_n(out, out_stride, q, scale, rows, cols, x, x_stride, 3u); break;
    case 2u: rows_i8_avx2_n(out, out_stride, q, scale, rows, cols, x, x_stride, 2u); break;
    default: rows_i8_avx2_n(out, out_stride, q, scale, rows, cols, x, x_stride, 1u); break;
    }
}

static void rows_i8_x86(float *restrict out, size_t out_stride,
                        const int8_t *restrict q, const float *restrict scale,
                        uint32_t rows, uint32_t cols, const float *restrict x,
                        size_t x_stride, uint32_t n);
#endif

void blink_matmul_i8(float *restrict out, const blink_tensor *w,
                     const float *restrict x)
{
    const int8_t *restrict q = (const int8_t *)w->data;
    const float *restrict scale = w->scale;
    const uint32_t rows = w->rows;
    const uint32_t cols = w->cols;

#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
    if (w->dense) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (__LAPACK_int)rows,
                    (__LAPACK_int)cols, 1.0f, w->dense, (__LAPACK_int)cols, x,
                    1, 0.0f, out, 1);
        return;
    }
#endif
#ifdef BLINK_SIMD_X86
    if (blink_x86_level() > 0) {
        rows_i8_x86(out, 1u, q, scale, rows, cols, x, 0u, 1u);
        return;
    }
#endif
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *restrict row = q + (size_t)r * cols;
#ifdef BLINK_SIMD_NEON
        out[r] = dot_i8_neon(row, x, cols) * scale[r];
#else
        out[r] = dot_i8_scalar(row, x, cols) * scale[r];
#endif
    }
}

/* The same product for up to BLINK_ROW_BLOCK input vectors at once.
 *
 * Converting int8 weights to fp32 is the expensive part of a W8A32 product,
 * and blink_matmul_i8 pays it once per weight per input vector. Here each
 * weight row is loaded and widened once and then multiplied into every input
 * vector of the group. Each vector keeps its own accumulators, updated in
 * exactly the order dot_i8_neon (or dot_i8_scalar) would update them, so every
 * output is bit-identical to calling blink_matmul_i8 once per vector: this
 * changes how often a weight is converted, not what is summed or in what
 * order. tests/c/test_kernels.c checks the equality with memcmp.
 *
 * `n` is a compile-time constant at every call site below, so the compiler
 * keeps the 4 * n accumulators in registers. */
#ifdef BLINK_SIMD_NEON
static inline __attribute__((always_inline)) void
rows_i8_neon(float *restrict out, size_t out_stride, const int8_t *restrict q,
             const float *restrict scale, uint32_t rows, uint32_t cols,
             const float *restrict x, size_t x_stride, uint32_t n)
{
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *restrict row = q + (size_t)r * cols;
        float32x4_t acc[BLINK_ROW_BLOCK][4];
        for (uint32_t v = 0; v < n; ++v) {
            for (uint32_t k = 0; k < 4u; ++k) {
                acc[v][k] = vdupq_n_f32(0.0f);
            }
        }
        uint32_t c = 0;
        for (; c + 16u <= cols; c += 16u) {
            const int8x16_t bytes = vld1q_s8(row + c);
            const int16x8_t low = vmovl_s8(vget_low_s8(bytes));
            const int16x8_t high = vmovl_s8(vget_high_s8(bytes));
            const float32x4_t w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(low)));
            const float32x4_t w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(low)));
            const float32x4_t w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(high)));
            const float32x4_t w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(high)));
            for (uint32_t v = 0; v < n; ++v) {
                const float *restrict xv = x + v * x_stride + c;
                acc[v][0] = vfmaq_f32(acc[v][0], w0, vld1q_f32(xv));
                acc[v][1] = vfmaq_f32(acc[v][1], w1, vld1q_f32(xv + 4u));
                acc[v][2] = vfmaq_f32(acc[v][2], w2, vld1q_f32(xv + 8u));
                acc[v][3] = vfmaq_f32(acc[v][3], w3, vld1q_f32(xv + 12u));
            }
        }
        for (uint32_t v = 0; v < n; ++v) {
            const float *restrict xv = x + v * x_stride;
            float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc[v][0], acc[v][1]),
                                             vaddq_f32(acc[v][2], acc[v][3])));
            for (uint32_t t = c; t < cols; ++t) {
                sum += (float)row[t] * xv[t];
            }
            out[v * out_stride + r] = sum * scale[r];
        }
    }
}
#else
static inline void
rows_i8_scalar(float *restrict out, size_t out_stride, const int8_t *restrict q,
               const float *restrict scale, uint32_t rows, uint32_t cols,
               const float *restrict x, size_t x_stride, uint32_t n)
{
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *restrict row = q + (size_t)r * cols;
        float sum[BLINK_ROW_BLOCK];
        for (uint32_t v = 0; v < n; ++v) {
            sum[v] = 0.0f;
        }
        for (uint32_t c = 0; c < cols; ++c) {
            const float weight = (float)row[c];
            for (uint32_t v = 0; v < n; ++v) {
                sum[v] += weight * x[v * x_stride + c];
            }
        }
        for (uint32_t v = 0; v < n; ++v) {
            out[v * out_stride + r] = sum[v] * scale[r];
        }
    }
}
#endif

#ifdef BLINK_SIMD_X86
/* One entry point per path, `n` still a constant inside each. */
static void rows_i8_x86(float *restrict out, size_t out_stride,
                        const int8_t *restrict q, const float *restrict scale,
                        uint32_t rows, uint32_t cols, const float *restrict x,
                        size_t x_stride, uint32_t n)
{
    if (blink_x86_level() == 2) {
        rows_i8_avx2(out, out_stride, q, scale, rows, cols, x, x_stride, n);
        return;
    }
    switch (n) {
    case 4u: rows_i8_sse2(out, out_stride, q, scale, rows, cols, x, x_stride, 4u); break;
    case 3u: rows_i8_sse2(out, out_stride, q, scale, rows, cols, x, x_stride, 3u); break;
    case 2u: rows_i8_sse2(out, out_stride, q, scale, rows, cols, x, x_stride, 2u); break;
    default: rows_i8_sse2(out, out_stride, q, scale, rows, cols, x, x_stride, 1u); break;
    }
}
#endif

void blink_matmul_i8_rows(float *restrict out, size_t out_stride,
                          const blink_tensor *w, const float *restrict x,
                          size_t x_stride, uint32_t count)
{
    const int8_t *restrict q = (const int8_t *)w->data;
    const float *restrict scale = w->scale;
    const uint32_t rows = w->rows;
    const uint32_t cols = w->cols;
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
    /* out[v][r] = sum_c x[v][c] * dense[r][c]: one product for every row. */
    if (w->dense && count > 1u) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (__LAPACK_int)count, (__LAPACK_int)rows, (__LAPACK_int)cols,
                    1.0f, x, (__LAPACK_int)x_stride, w->dense,
                    (__LAPACK_int)cols, 0.0f, out, (__LAPACK_int)out_stride);
        return;
    }
#endif
#ifdef BLINK_SIMD_X86
    if (blink_x86_level() > 0) {
        /* Groups of four, then the rest as one group: the one-vector case is
         * the same kernel with n = 1, which is what blink_matmul_i8 runs. */
        uint32_t start = 0;
        for (; start < count; start += BLINK_ROW_BLOCK) {
            const uint32_t n = count - start < BLINK_ROW_BLOCK
                                   ? count - start : BLINK_ROW_BLOCK;
            rows_i8_x86(out + start * out_stride, out_stride, q, scale, rows,
                        cols, x + start * x_stride, x_stride, n);
        }
        return;
    }
#endif
#ifdef BLINK_SIMD_NEON
#define BLINK_ROWS_KERNEL rows_i8_neon
#else
#define BLINK_ROWS_KERNEL rows_i8_scalar
#endif
    uint32_t done = 0;
    for (; done + 4u <= count; done += 4u) {
        BLINK_ROWS_KERNEL(out + done * out_stride, out_stride, q, scale, rows,
                          cols, x + done * x_stride, x_stride, 4u);
    }
    switch (count - done) {
    case 3u:
        BLINK_ROWS_KERNEL(out + done * out_stride, out_stride, q, scale, rows,
                          cols, x + done * x_stride, x_stride, 3u);
        break;
    case 2u:
        BLINK_ROWS_KERNEL(out + done * out_stride, out_stride, q, scale, rows,
                          cols, x + done * x_stride, x_stride, 2u);
        break;
    case 1u:
        blink_matmul_i8(out + done * out_stride, w, x + done * x_stride);
        break;
    default:
        break;
    }
#undef BLINK_ROWS_KERNEL
}

/* ============================================================== W8A8 */

#if defined(BLINK_W8A8) && BLINK_W8A8

/* 0 scalar, 1 SDOT, 2 I8MM, 3 SSE2, 4 AVX2. Starts at the best kernel the
 * compiler alone guarantees; blink_kernels_init upgrades it when the CPU
 * reports more. */
#ifdef BLINK_HAVE_SDOT
static int w8a8_kernel = 1;
#elif defined(BLINK_SIMD_X86)
static int w8a8_kernel = 3;
#else
static int w8a8_kernel = 0;
#endif
static int w8a8_detected = 0;

static int cpu_has_i8mm(void)
{
#if defined(BLINK_HAVE_I8MM) && defined(__APPLE__)
    int value = 0;
    size_t size = sizeof value;
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &value, &size, NULL, 0) != 0) {
        return 0;
    }
    return value != 0;
#elif defined(BLINK_HAVE_I8MM) && defined(__linux__)
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
    return 0;
#endif
}

/* I8MM is chosen automatically where it has a reason to be: not on Apple
 * silicon, where it measured level with SDOT (docs/RESULTS.md, *The W8A8
 * backend*) and would cost a rearranged copy of the projections for nothing.
 * BLINK_W8A8_KERNEL=scalar|sdot|i8mm overrides the choice, for measuring one
 * kernel against another; a kernel the CPU or the build lacks is ignored.
 * Read once, when the first model opens. */
void blink_kernels_init(void)
{
    if (w8a8_detected) {
        return;
    }
    w8a8_detected = 1;
#if !defined(__APPLE__)
    if (cpu_has_i8mm()) {
        w8a8_kernel = 2;
    }
#endif
#ifdef BLINK_SIMD_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) {
        w8a8_kernel = 4;
    }
#endif
    const char *requested = getenv("BLINK_W8A8_KERNEL");
    if (requested) {
        if (strcmp(requested, "scalar") == 0) {
            blink_w8a8_force_kernel(0);
        } else if (strcmp(requested, "sdot") == 0) {
            blink_w8a8_force_kernel(1);
        } else if (strcmp(requested, "i8mm") == 0) {
            blink_w8a8_force_kernel(2);
        } else if (strcmp(requested, "sse2") == 0) {
            blink_w8a8_force_kernel(3);
        } else if (strcmp(requested, "avx2") == 0) {
            blink_w8a8_force_kernel(4);
        }
    }
}

int blink_w8a8_kernel(void)
{
    return w8a8_kernel;
}

/* For tests/c/test_kernels.c: run a specific kernel. Asking for one the CPU
 * or the build lacks leaves the current choice alone. Returns the kernel now
 * in use. */
int blink_w8a8_force_kernel(int kernel)
{
    if (kernel == 0) {
        w8a8_kernel = 0;
    }
#ifdef BLINK_HAVE_SDOT
    if (kernel == 1) {
        w8a8_kernel = 1;
    }
#endif
    if (kernel == 2 && cpu_has_i8mm()) {
        w8a8_kernel = 2;
    }
#ifdef BLINK_SIMD_X86
    if (kernel == 3) {
        w8a8_kernel = 3;
    }
    if (kernel == 4 && __builtin_cpu_supports("avx2")) {
        w8a8_kernel = 4;
    }
#endif
    return w8a8_kernel;
}

static int8_t quantize_one(float value, float inverse)
{
    long q = lrintf(value * inverse);
    if (q > 127L) q = 127L;
    if (q < -127L) q = -127L;
    return (int8_t)q;
}

/* The NEON path computes the same maximum (exact, in any order) and the same
 * products, and vcvtnq rounds half to even as lrintf does in the default
 * rounding mode; |value * inverse| never exceeds 127 by more than an ulp, so
 * the saturating narrow and the clamp agree. */
float blink_quantize_row(int8_t *restrict q, const float *restrict x,
                         uint32_t n)
{
    float maximum = 0.0f;
    uint32_t i = 0;
#ifdef BLINK_SIMD_NEON
    float32x4_t m4 = vdupq_n_f32(0.0f);
    for (; i + 4u <= n; i += 4u) {
        m4 = vmaxq_f32(m4, vabsq_f32(vld1q_f32(x + i)));
    }
    maximum = vmaxvq_f32(m4);
#elif defined(BLINK_SIMD_X86)
    const __m128 magnitude_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    __m128 m4 = _mm_setzero_ps();
    for (; i + 4u <= n; i += 4u) {
        m4 = _mm_max_ps(m4, _mm_and_ps(_mm_loadu_ps(x + i), magnitude_mask));
    }
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    maximum = _mm_cvtss_f32(m4);
#endif
    for (; i < n; ++i) {
        const float magnitude = fabsf(x[i]);
        if (magnitude > maximum) {
            maximum = magnitude;
        }
    }
    if (!(maximum > 0.0f)) {
        memset(q, 0, n);
        return 0.0f;
    }
    const float inverse = 127.0f / maximum;
    i = 0;
#ifdef BLINK_SIMD_NEON
    const float32x4_t inverse4 = vdupq_n_f32(inverse);
    for (; i + 8u <= n; i += 8u) {
        const int32x4_t lo = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i), inverse4));
        const int32x4_t hi =
            vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4u), inverse4));
        vst1_s8(q + i, vqmovn_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))));
    }
#elif defined(BLINK_SIMD_X86)
    /* cvtps2dq rounds by MXCSR, round half to even unless the caller changed
     * it, which is also what lrintf does; the saturating packs agree with the
     * clamp for the same reason as the NEON narrow. */
    const __m128 inverse4 = _mm_set1_ps(inverse);
    for (; i + 8u <= n; i += 8u) {
        const __m128i lo = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(x + i), inverse4));
        const __m128i hi =
            _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(x + i + 4u), inverse4));
        const __m128i packed = _mm_packs_epi16(_mm_packs_epi32(lo, hi),
                                               _mm_setzero_si128());
        _mm_storel_epi64((__m128i *)(q + i), packed);
    }
#endif
    for (; i < n; ++i) {
        q[i] = quantize_one(x[i], inverse);
    }
    return maximum / 127.0f;
}

/* The normative integer product. */
static int32_t dot_s8_scalar(const int8_t *restrict a, const int8_t *restrict b,
                             uint32_t n)
{
    int32_t sum = 0;
    for (uint32_t c = 0; c < n; ++c) {
        sum += (int32_t)a[c] * (int32_t)b[c];
    }
    return sum;
}

static float w8a8_output(int32_t sum, float row_scale, float x_scale)
{
    return ((float)sum * row_scale) * x_scale;
}

static void w8a8_scalar(float *restrict out, size_t out_stride,
                        const int8_t *restrict wq, const float *restrict scale,
                        uint32_t rows, uint32_t cols,
                        const int8_t *restrict xq, const float *restrict sx,
                        uint32_t n)
{
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *row = wq + (size_t)r * cols;
        for (uint32_t v = 0; v < n; ++v) {
            out[v * out_stride + r] = w8a8_output(
                dot_s8_scalar(row, xq + (size_t)v * cols, cols), scale[r], sx[v]);
        }
    }
}

#ifdef BLINK_HAVE_SDOT
/* Four weight rows against `n` input rows, sixteen columns per SDOT. Each
 * (input, weight) pair has its own accumulator, and one pairwise reduction
 * turns four of them into the four outputs of a row group. `n` is a
 * compile-time constant at every call, so the 4 * n accumulators stay in
 * registers. */
static inline __attribute__((always_inline)) void
w8a8_sdot_n(float *restrict out, size_t out_stride, const int8_t *restrict wq,
            const float *restrict scale, uint32_t rows, uint32_t cols,
            const int8_t *restrict xq, const float *restrict sx, uint32_t n)
{
    const uint32_t body = cols & ~15u;
    uint32_t r = 0;
    for (; r + 4u <= rows; r += 4u) {
        const int8_t *w0 = wq + (size_t)r * cols;
        int32x4_t acc[BLINK_ROW_BLOCK][4];
        for (uint32_t v = 0; v < n; ++v) {
            for (uint32_t k = 0; k < 4u; ++k) acc[v][k] = vdupq_n_s32(0);
        }
        for (uint32_t c = 0; c < body; c += 16u) {
            const int8x16_t a0 = vld1q_s8(w0 + c);
            const int8x16_t a1 = vld1q_s8(w0 + cols + c);
            const int8x16_t a2 = vld1q_s8(w0 + 2u * cols + c);
            const int8x16_t a3 = vld1q_s8(w0 + 3u * cols + c);
            for (uint32_t v = 0; v < n; ++v) {
                const int8x16_t in = vld1q_s8(xq + (size_t)v * cols + c);
                acc[v][0] = vdotq_s32(acc[v][0], a0, in);
                acc[v][1] = vdotq_s32(acc[v][1], a1, in);
                acc[v][2] = vdotq_s32(acc[v][2], a2, in);
                acc[v][3] = vdotq_s32(acc[v][3], a3, in);
            }
        }
        for (uint32_t v = 0; v < n; ++v) {
            const int32x4_t sums =
                vpaddq_s32(vpaddq_s32(acc[v][0], acc[v][1]),
                           vpaddq_s32(acc[v][2], acc[v][3]));
            const int8_t *xv = xq + (size_t)v * cols;
            for (uint32_t k = 0; k < 4u; ++k) {
                int32_t sum = sums[k];
                if (body != cols) {
                    sum += dot_s8_scalar(w0 + (size_t)k * cols + body,
                                         xv + body, cols - body);
                }
                out[v * out_stride + r + k] =
                    w8a8_output(sum, scale[r + k], sx[v]);
            }
        }
    }
    if (r < rows) {
        w8a8_scalar(out + r, out_stride, wq + (size_t)r * cols, scale + r,
                    rows - r, cols, xq, sx, n);
    }
}

static void w8a8_sdot(float *restrict out, size_t out_stride,
                      const int8_t *restrict wq, const float *restrict scale,
                      uint32_t rows, uint32_t cols,
                      const int8_t *restrict xq, const float *restrict sx,
                      uint32_t n)
{
    switch (n) {
    case 4u: w8a8_sdot_n(out, out_stride, wq, scale, rows, cols, xq, sx, 4u); break;
    case 3u: w8a8_sdot_n(out, out_stride, wq, scale, rows, cols, xq, sx, 3u); break;
    case 2u: w8a8_sdot_n(out, out_stride, wq, scale, rows, cols, xq, sx, 2u); break;
    default: w8a8_sdot_n(out, out_stride, wq, scale, rows, cols, xq, sx, 1u); break;
    }
}
#endif

#ifdef BLINK_SIMD_X86
static inline __attribute__((always_inline)) int32_t hsum_epi32(__m128i v)
{
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v);
}

/* SSE2: sixteen bytes of weights and inputs sign-extended to int16 and
 * multiplied pairwise into int32 by pmaddwd. Exact, like every W8A8 kernel. */
static inline __attribute__((always_inline)) void
w8a8_sse2_n(float *restrict out, size_t out_stride, const int8_t *restrict wq,
            const float *restrict scale, uint32_t rows, uint32_t cols,
            const int8_t *restrict xq, const float *restrict sx, uint32_t n)
{
    const uint32_t body = cols & ~15u;
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *row = wq + (size_t)r * cols;
        __m128i acc[BLINK_ROW_BLOCK];
        for (uint32_t v = 0; v < n; ++v) acc[v] = _mm_setzero_si128();
        for (uint32_t c = 0; c < body; c += 16u) {
            const __m128i w = _mm_loadu_si128((const __m128i *)(row + c));
            const __m128i wl = _mm_srai_epi16(_mm_unpacklo_epi8(w, w), 8);
            const __m128i wh = _mm_srai_epi16(_mm_unpackhi_epi8(w, w), 8);
            for (uint32_t v = 0; v < n; ++v) {
                const __m128i in =
                    _mm_loadu_si128((const __m128i *)(xq + (size_t)v * cols + c));
                const __m128i il = _mm_srai_epi16(_mm_unpacklo_epi8(in, in), 8);
                const __m128i ih = _mm_srai_epi16(_mm_unpackhi_epi8(in, in), 8);
                acc[v] = _mm_add_epi32(acc[v], _mm_madd_epi16(wl, il));
                acc[v] = _mm_add_epi32(acc[v], _mm_madd_epi16(wh, ih));
            }
        }
        for (uint32_t v = 0; v < n; ++v) {
            int32_t sum = hsum_epi32(acc[v]);
            if (body != cols) {
                sum += dot_s8_scalar(row + body, xq + (size_t)v * cols + body,
                                     cols - body);
            }
            out[v * out_stride + r] = w8a8_output(sum, scale[r], sx[v]);
        }
    }
}

static void w8a8_sse2(float *restrict out, size_t out_stride,
                      const int8_t *restrict wq, const float *restrict scale,
                      uint32_t rows, uint32_t cols,
                      const int8_t *restrict xq, const float *restrict sx,
                      uint32_t n)
{
    switch (n) {
    case 4u: w8a8_sse2_n(out, out_stride, wq, scale, rows, cols, xq, sx, 4u); break;
    case 3u: w8a8_sse2_n(out, out_stride, wq, scale, rows, cols, xq, sx, 3u); break;
    case 2u: w8a8_sse2_n(out, out_stride, wq, scale, rows, cols, xq, sx, 2u); break;
    default: w8a8_sse2_n(out, out_stride, wq, scale, rows, cols, xq, sx, 1u); break;
    }
}

/* AVX2 has no signed-by-signed byte product, so the sign moves to the other
 * operand: |w| times (x with w's sign) is w times x. vpmaddubsw multiplies
 * unsigned by signed bytes and adds neighbours into int16; with |w| <= 128
 * and |x| <= 127 a pair is at most 32,512, so it never saturates, and
 * vpmaddwd by ones widens to int32. The sum stays exact. */
BLINK_AVX2_TARGET
static inline __attribute__((always_inline)) void
w8a8_avx2_n(float *restrict out, size_t out_stride, const int8_t *restrict wq,
            const float *restrict scale, uint32_t rows, uint32_t cols,
            const int8_t *restrict xq, const float *restrict sx, uint32_t n)
{
    const uint32_t body = cols & ~31u;
    const __m256i ones = _mm256_set1_epi16(1);
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *row = wq + (size_t)r * cols;
        __m256i acc[BLINK_ROW_BLOCK];
        for (uint32_t v = 0; v < n; ++v) acc[v] = _mm256_setzero_si256();
        for (uint32_t c = 0; c < body; c += 32u) {
            const __m256i w = _mm256_loadu_si256((const __m256i *)(row + c));
            const __m256i magnitude = _mm256_sign_epi8(w, w);
            for (uint32_t v = 0; v < n; ++v) {
                const __m256i in = _mm256_loadu_si256(
                    (const __m256i *)(xq + (size_t)v * cols + c));
                const __m256i pairs =
                    _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(in, w));
                acc[v] = _mm256_add_epi32(acc[v], _mm256_madd_epi16(pairs, ones));
            }
        }
        for (uint32_t v = 0; v < n; ++v) {
            int32_t sum = hsum_epi32(_mm_add_epi32(_mm256_castsi256_si128(acc[v]),
                                                   _mm256_extracti128_si256(acc[v], 1)));
            if (body != cols) {
                sum += dot_s8_scalar(row + body, xq + (size_t)v * cols + body,
                                     cols - body);
            }
            out[v * out_stride + r] = w8a8_output(sum, scale[r], sx[v]);
        }
    }
}

BLINK_AVX2_TARGET
static void w8a8_avx2(float *restrict out, size_t out_stride,
                      const int8_t *restrict wq, const float *restrict scale,
                      uint32_t rows, uint32_t cols,
                      const int8_t *restrict xq, const float *restrict sx,
                      uint32_t n)
{
    switch (n) {
    case 4u: w8a8_avx2_n(out, out_stride, wq, scale, rows, cols, xq, sx, 4u); break;
    case 3u: w8a8_avx2_n(out, out_stride, wq, scale, rows, cols, xq, sx, 3u); break;
    case 2u: w8a8_avx2_n(out, out_stride, wq, scale, rows, cols, xq, sx, 2u); break;
    default: w8a8_avx2_n(out, out_stride, wq, scale, rows, cols, xq, sx, 1u); break;
    }
}
#endif

size_t blink_packed_i8mm_bytes(uint32_t rows, uint32_t cols)
{
    return (size_t)(rows & ~3u) * (cols & ~15u);
}

void blink_pack_i8mm(int8_t *restrict packed, const int8_t *restrict q,
                     uint32_t rows, uint32_t cols)
{
    const uint32_t body = cols & ~15u;
    for (uint32_t r = 0; r + 4u <= rows; r += 4u) {
        for (uint32_t c = 0; c < body; c += 16u) {
            /* operands: rows (0,1) low, (0,1) high, (2,3) low, (2,3) high */
            for (uint32_t op = 0; op < 4u; ++op) {
                const uint32_t pair = op / 2u, half = op % 2u;
                for (uint32_t i = 0; i < 16u; ++i) {
                    const uint32_t row = r + 2u * pair + i / 8u;
                    *packed++ = q[(size_t)row * cols + c + 8u * half + i % 8u];
                }
            }
        }
    }
}

#ifdef BLINK_HAVE_I8MM
/* What the I8MM path hands an unpaired input row, or a tensor the model did
 * not pack, to: the same integer sums by another route. */
#ifdef BLINK_HAVE_SDOT
#define W8A8_FALLBACK w8a8_sdot
#else
#define W8A8_FALLBACK w8a8_scalar
#endif

/* The input rows of a group in the same layout: for each sixteen columns and
 * each pair of rows, the two 2x8 operands (low halves, high halves). */
static void pack_inputs(int8_t *restrict packed, const int8_t *restrict xq,
                        uint32_t cols, uint32_t pairs)
{
    const uint32_t body = cols & ~15u;
    for (uint32_t c = 0; c < body; c += 16u) {
        for (uint32_t p = 0; p < pairs; ++p) {
            const int8_t *x0 = xq + (size_t)(2u * p) * cols;
            const int64x2_t a = vreinterpretq_s64_s8(vld1q_s8(x0 + c));
            const int64x2_t b = vreinterpretq_s64_s8(vld1q_s8(x0 + cols + c));
            vst1q_s8(packed, vreinterpretq_s8_s64(vzip1q_s64(a, b)));
            vst1q_s8(packed + 16u, vreinterpretq_s8_s64(vzip2q_s64(a, b)));
            packed += 32u;
        }
    }
}

/* One SMMLA multiplies a 2x8 block of input rows by a 2x8 block of weight
 * rows into a 2x2 block of sums, 32 products per instruction; lanes are
 * (x0.w0, x0.w1, x1.w0, x1.w1). Both operands come pre-arranged, the weights
 * from the model and the inputs from pack_inputs, so the loop is loads and
 * SMMLAs only. The tile is four weight rows by `pairs` pairs of input rows;
 * columns past a multiple of sixteen and rows past a multiple of four use the
 * scalar product on the original layout. */
BLINK_I8MM_TARGET
static inline __attribute__((always_inline)) void
w8a8_i8mm_n(float *restrict out, size_t out_stride, const blink_tensor *w,
            const int8_t *restrict xq, const int8_t *restrict xp,
            const float *restrict sx, uint32_t pairs)
{
    const int8_t *wq = (const int8_t *)w->data;
    const float *scale = w->scale;
    const uint32_t rows = w->rows, cols = w->cols;
    const uint32_t body = cols & ~15u;
    const uint32_t chunks = body / 16u;
    uint32_t r = 0;
    /* Eight weight rows at a time: 2 * pairs accumulators per four rows, each
     * holding four outputs, so the wider tile still fits in registers and
     * loads 12 vectors per 512 products instead of 8 per 256. */
    for (; r + 8u <= rows; r += 8u) {
        const int8_t *wa = w->packed + (size_t)r * body;
        const int8_t *wb = wa + (size_t)4u * body;
        int32x4_t acc[BLINK_ROW_BLOCK / 2u][4];
        for (uint32_t p = 0; p < pairs; ++p) {
            for (uint32_t q = 0; q < 4u; ++q) acc[p][q] = vdupq_n_s32(0);
        }
        for (uint32_t j = 0; j < chunks; ++j) {
            const int8x16_t a01lo = vld1q_s8(wa + 64u * j);
            const int8x16_t a01hi = vld1q_s8(wa + 64u * j + 16u);
            const int8x16_t a23lo = vld1q_s8(wa + 64u * j + 32u);
            const int8x16_t a23hi = vld1q_s8(wa + 64u * j + 48u);
            const int8x16_t b01lo = vld1q_s8(wb + 64u * j);
            const int8x16_t b01hi = vld1q_s8(wb + 64u * j + 16u);
            const int8x16_t b23lo = vld1q_s8(wb + 64u * j + 32u);
            const int8x16_t b23hi = vld1q_s8(wb + 64u * j + 48u);
            for (uint32_t p = 0; p < pairs; ++p) {
                const int8_t *x = xp + (size_t)(j * pairs + p) * 32u;
                const int8x16_t xlo = vld1q_s8(x);
                const int8x16_t xhi = vld1q_s8(x + 16u);
                acc[p][0] = vmmlaq_s32(acc[p][0], xlo, a01lo);
                acc[p][0] = vmmlaq_s32(acc[p][0], xhi, a01hi);
                acc[p][1] = vmmlaq_s32(acc[p][1], xlo, a23lo);
                acc[p][1] = vmmlaq_s32(acc[p][1], xhi, a23hi);
                acc[p][2] = vmmlaq_s32(acc[p][2], xlo, b01lo);
                acc[p][2] = vmmlaq_s32(acc[p][2], xhi, b01hi);
                acc[p][3] = vmmlaq_s32(acc[p][3], xlo, b23lo);
                acc[p][3] = vmmlaq_s32(acc[p][3], xhi, b23hi);
            }
        }
        for (uint32_t p = 0; p < pairs; ++p) {
            for (uint32_t h = 0; h < 4u; ++h) {
                for (uint32_t lane = 0; lane < 4u; ++lane) {
                    const uint32_t v = 2u * p + lane / 2u;
                    const uint32_t k = 2u * h + lane % 2u;
                    int32_t sum = acc[p][h][lane];
                    if (body != cols) {
                        sum += dot_s8_scalar(wq + (size_t)(r + k) * cols + body,
                                             xq + (size_t)v * cols + body,
                                             cols - body);
                    }
                    out[v * out_stride + r + k] =
                        w8a8_output(sum, scale[r + k], sx[v]);
                }
            }
        }
    }
    for (; r + 4u <= rows; r += 4u) {
        const int8_t *wp = w->packed + (size_t)r * body;
        int32x4_t acc[BLINK_ROW_BLOCK / 2u][2];
        for (uint32_t p = 0; p < pairs; ++p) {
            acc[p][0] = vdupq_n_s32(0);
            acc[p][1] = vdupq_n_s32(0);
        }
        for (uint32_t j = 0; j < chunks; ++j) {
            const int8x16_t w01lo = vld1q_s8(wp + 64u * j);
            const int8x16_t w01hi = vld1q_s8(wp + 64u * j + 16u);
            const int8x16_t w23lo = vld1q_s8(wp + 64u * j + 32u);
            const int8x16_t w23hi = vld1q_s8(wp + 64u * j + 48u);
            for (uint32_t p = 0; p < pairs; ++p) {
                const int8_t *x = xp + (size_t)(j * pairs + p) * 32u;
                const int8x16_t xlo = vld1q_s8(x);
                const int8x16_t xhi = vld1q_s8(x + 16u);
                acc[p][0] = vmmlaq_s32(acc[p][0], xlo, w01lo);
                acc[p][0] = vmmlaq_s32(acc[p][0], xhi, w01hi);
                acc[p][1] = vmmlaq_s32(acc[p][1], xlo, w23lo);
                acc[p][1] = vmmlaq_s32(acc[p][1], xhi, w23hi);
            }
        }
        for (uint32_t p = 0; p < pairs; ++p) {
            for (uint32_t h = 0; h < 2u; ++h) {
                for (uint32_t lane = 0; lane < 4u; ++lane) {
                    const uint32_t v = 2u * p + lane / 2u;
                    const uint32_t k = 2u * h + lane % 2u;
                    int32_t sum = acc[p][h][lane];
                    if (body != cols) {
                        sum += dot_s8_scalar(wq + (size_t)(r + k) * cols + body,
                                             xq + (size_t)v * cols + body,
                                             cols - body);
                    }
                    out[v * out_stride + r + k] =
                        w8a8_output(sum, scale[r + k], sx[v]);
                }
            }
        }
    }
    if (r < rows) {
        w8a8_scalar(out + r, out_stride, wq + (size_t)r * cols, scale + r,
                    rows - r, cols, xq, sx, 2u * pairs);
    }
}

BLINK_I8MM_TARGET
static void w8a8_i8mm(float *restrict out, size_t out_stride,
                      const blink_tensor *w, const int8_t *restrict xq,
                      int8_t *restrict xp, const float *restrict sx, uint32_t n)
{
    const uint32_t pairs = n / 2u;
    if (pairs) {
        pack_inputs(xp, xq, w->cols, pairs);
        if (pairs == 2u) {
            w8a8_i8mm_n(out, out_stride, w, xq, xp, sx, 2u);
        } else {
            w8a8_i8mm_n(out, out_stride, w, xq, xp, sx, 1u);
        }
    }
    if (n % 2u) {
        const uint32_t v = n - 1u;
        W8A8_FALLBACK(out + (size_t)v * out_stride, out_stride,
                      (const int8_t *)w->data, w->scale, w->rows, w->cols,
                      xq + (size_t)v * w->cols, sx + v, 1u);
    }
}
#endif

void blink_matmul_w8a8_rows(float *restrict out, size_t out_stride,
                            const blink_tensor *w, const float *restrict x,
                            size_t x_stride, uint32_t count,
                            int8_t *restrict scratch)
{
    const int8_t *wq = (const int8_t *)w->data;
    const uint32_t rows = w->rows;
    const uint32_t cols = w->cols;
    float sx[BLINK_ROW_BLOCK];

    for (uint32_t done = 0; done < count;) {
        const uint32_t n =
            count - done < BLINK_ROW_BLOCK ? count - done : BLINK_ROW_BLOCK;
        for (uint32_t v = 0; v < n; ++v) {
            sx[v] = blink_quantize_row(scratch + (size_t)v * cols,
                                       x + (size_t)(done + v) * x_stride, cols);
        }
        float *o = out + (size_t)done * out_stride;
        switch (w8a8_kernel) {
#ifdef BLINK_HAVE_I8MM
        case 2:
            if (w->packed) {
                w8a8_i8mm(o, out_stride, w, scratch,
                          scratch + (size_t)BLINK_ROW_BLOCK * cols, sx, n);
                break;
            }
            /* a tensor the model did not pack: the same sums another way */
            W8A8_FALLBACK(o, out_stride, wq, w->scale, rows, cols, scratch, sx,
                          n);
            break;
#endif
#ifdef BLINK_SIMD_X86
        case 4:
            w8a8_avx2(o, out_stride, wq, w->scale, rows, cols, scratch, sx, n);
            break;
        case 3:
            w8a8_sse2(o, out_stride, wq, w->scale, rows, cols, scratch, sx, n);
            break;
#endif
#ifdef BLINK_HAVE_SDOT
        case 1:
            w8a8_sdot(o, out_stride, wq, w->scale, rows, cols, scratch, sx, n);
            break;
#endif
        default:
            w8a8_scalar(o, out_stride, wq, w->scale, rows, cols, scratch, sx, n);
            break;
        }
        done += n;
    }
}

#else

void blink_kernels_init(void) {}
int blink_w8a8_kernel(void) { return 0; }

#endif /* BLINK_W8A8 */

const char *blink_backend(void)
{
#if defined(BLINK_W8A8) && BLINK_W8A8
    blink_kernels_init();
    switch (w8a8_kernel) {
    case 4: return "w8a8-avx2";
    case 3: return "w8a8-sse2";
    case 2: return "w8a8-i8mm";
    case 1: return "w8a8-sdot";
    default: return "w8a8-scalar";
    }
#elif defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
    return "accelerate";
#elif defined(BLINK_SIMD_NEON)
    return "neon";
#elif defined(BLINK_SIMD_X86)
    switch (blink_x86_level()) {
    case 2: return "avx2";
    case 1: return "sse2";
    default: return "scalar";
    }
#else
    return "scalar";
#endif
}

/* Exposed so tests/c/test_kernels.c can compare the two paths directly. */
float blink_dot_i8_scalar(const int8_t *row, const float *x, uint32_t cols)
{
    return dot_i8_scalar(row, x, cols);
}

#if !defined(BLINK_SIMD_X86)
/* No x86 SIMD in this build (another architecture, or BLINK_SCALAR_ONLY):
 * the scalar loop is the only level. */
int blink_x86_level(void) { return 0; }
int blink_x86_force_level(int level) { (void)level; return 0; }
#endif

int blink_accelerate_enabled(void)
{
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
    return 1;
#else
    return 0;
#endif
}

int blink_simd_enabled(void)
{
#ifdef BLINK_SIMD_NEON
    return 1;
#elif defined(BLINK_SIMD_X86)
    return blink_x86_level() > 0;
#else
    return 0;
#endif
}

/* y[r] = sum_j w[r][j] * x[j] over a dense fp32 matrix. Used only by the FiLM
 * modulation, whose weights stay fp32 because they scale the option vectors:
 * see the note in python/blink_train/export.py. */
void blink_matmul_f32(float *restrict out, const float *restrict w,
                      const float *restrict x, uint32_t rows, uint32_t cols)
{
    for (uint32_t r = 0; r < rows; ++r) {
        out[r] = blink_dot(w + (size_t)r * cols, x, cols);
    }
}

#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
/* The two halves of one attention read as matrix-vector products; the
 * default build uses the inline loops in blink_internal.h. */
void blink_attention_scores(float *restrict scores, const float *restrict keys,
                            size_t key_stride, const float *restrict query,
                            uint32_t dim, uint32_t count, float scale)
{
    if (count == 0u) {
        return;
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (__LAPACK_int)count,
                (__LAPACK_int)dim, scale, keys, (__LAPACK_int)key_stride, query,
                1, 0.0f, scores, 1);
}

void blink_attention_accum(float *restrict out, const float *restrict values,
                           size_t value_stride, const float *restrict weights,
                           uint32_t dim, uint32_t count)
{
    if (count == 0u) {
        return;
    }
    cblas_sgemv(CblasRowMajor, CblasTrans, (__LAPACK_int)count,
                (__LAPACK_int)dim, 1.0f, values, (__LAPACK_int)value_stride,
                weights, 1, 1.0f, out, 1);
}
#endif

void blink_dwconv_accum(float *restrict out, const float *restrict in,
                        const float *restrict weight, uint32_t length,
                        uint32_t width, uint32_t kernel)
{
    const int32_t half = (int32_t)(kernel / 2u);
    for (uint32_t t = 0; t < length; ++t) {
        float *restrict dst = out + (size_t)t * width;
        for (uint32_t k = 0; k < kernel; ++k) {
            const int32_t source = (int32_t)t + (int32_t)k - half;
            if (source < 0 || source >= (int32_t)length) {
                continue;
            }
            const float *restrict src = in + (size_t)source * width;
            const float *restrict wk = weight + (size_t)k * width;
            for (uint32_t c = 0; c < width; ++c) {
                dst[c] += wk[c] * src[c];
            }
        }
    }
}

uint32_t blink_pool(float *restrict data, uint32_t length, uint32_t width,
                    uint32_t stride)
{
    if (stride <= 1u || length == 0u) {
        return length;
    }
    const uint32_t positions = (length + stride - 1u) / stride;
    for (uint32_t p = 0; p < positions; ++p) {
        const uint32_t start = p * stride;
        uint32_t stop = start + stride;
        if (stop > length) {
            stop = length;
        }
        float *restrict dst = data + (size_t)p * width;
        /* The first source row of window p is row p*stride >= p, so writing
         * the window mean back over row p never clobbers a row still needed. */
        if (start != p) {
            for (uint32_t c = 0; c < width; ++c) {
                dst[c] = data[(size_t)start * width + c];
            }
        }
        for (uint32_t t = start + 1u; t < stop; ++t) {
            const float *restrict src = data + (size_t)t * width;
            for (uint32_t c = 0; c < width; ++c) {
                dst[c] += src[c];
            }
        }
        const float inverse = 1.0f / (float)(stop - start);
        for (uint32_t c = 0; c < width; ++c) {
            dst[c] *= inverse;
        }
    }
    return positions;
}

/* Scale a vector to unit length in place, leaving a zero vector alone.
 *
 * The option head uses this on its queries, its keys and the attended vector,
 * which bounds both the attention scores and the final logit no matter how
 * large the projections grow. Without it, a run can increase |q|, |k| and |v|
 * to sharpen attention and raise confidence, and nothing stops the feedback:
 * one blink-small run reached a validation NLL of 160 that way. */
void blink_l2_normalize(float *restrict values, uint32_t count)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        sum += values[i] * values[i];
    }
    if (sum <= 0.0f) {
        return;
    }
    const float inverse = 1.0f / sqrtf(sum);
    for (uint32_t i = 0; i < count; ++i) {
        values[i] *= inverse;
    }
}

void blink_softmax(float *restrict values, uint32_t count)
{
    float maximum = values[0];
    for (uint32_t i = 1; i < count; ++i) {
        if (values[i] > maximum) {
            maximum = values[i];
        }
    }
    float total = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        const float weight = expf(values[i] - maximum);
        values[i] = weight;
        total += weight;
    }
    const float inv = 1.0f / total;
    for (uint32_t i = 0; i < count; ++i) {
        values[i] *= inv;
    }
}

/* Two rounds of a 32-bit integer mix. The Python trainer, the exporter and the
 * runtime must all produce identical bucket indices, so this is written with
 * explicit 32-bit wrapping and no platform-dependent types. */
uint32_t blink_bigram_index(uint8_t previous, uint8_t current,
                            uint32_t buckets)
{
    uint32_t h = (uint32_t)previous * 0x9E3779B1u;
    h ^= (uint32_t)current * 0x85EBCA77u;
    h ^= h >> 15;
    h = h * 0x2545F491u;
    h ^= h >> 13;
    return h & (buckets - 1u); /* buckets is validated as a power of two */
}
