/* Unit tests for the numeric kernels and the checksum.
 * SPDX-License-Identifier: Apache-2.0 */

#include "blink_internal.h"
#include "harness.h"

static void test_layernorm(void)
{
    float gain[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float bias[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];

    blink_layernorm(out, in, gain, bias, 4);

    /* mean 2.5, population variance 1.25, sigma = sqrt(1.25 + 1e-5) */
    const double sigma = sqrt(1.25 + 1e-5);
    for (int i = 0; i < 4; ++i) {
        CHECK_NEAR(out[i], ((double)in[i] - 2.5) / sigma, 1e-5);
    }

    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += out[i];
    }
    CHECK_NEAR(sum, 0.0, 1e-5);

    /* gain and bias are applied after normalisation */
    for (int i = 0; i < 4; ++i) {
        gain[i] = 2.0f;
        bias[i] = -1.0f;
    }
    float scaled[4];
    blink_layernorm(scaled, in, gain, bias, 4);
    for (int i = 0; i < 4; ++i) {
        CHECK_NEAR(scaled[i], 2.0 * out[i] - 1.0, 1e-5);
    }

    /* a constant vector normalises to the bias, without dividing by zero */
    float flat[4] = {7.0f, 7.0f, 7.0f, 7.0f};
    blink_layernorm(out, flat, gain, bias, 4);
    for (int i = 0; i < 4; ++i) {
        CHECK_NEAR(out[i], -1.0, 1e-5);
    }
}

static void test_softmax(void)
{
    float values[3] = {1.0f, 2.0f, 3.0f};
    blink_softmax(values, 3);
    double total = 0.0;
    for (int i = 0; i < 3; ++i) {
        CHECK(values[i] > 0.0f);
        total += values[i];
    }
    CHECK_NEAR(total, 1.0, 1e-6);
    CHECK_NEAR(values[2] / values[1], exp(1.0), 1e-5);

    /* large inputs must not overflow: the kernel subtracts the maximum */
    float large[2] = {200.0f, 100.0f};
    blink_softmax(large, 2);
    CHECK_NEAR(large[0], 1.0, 1e-6);
    CHECK_NEAR(large[1], 0.0, 1e-6);

    float uniform[5] = {4.0f, 4.0f, 4.0f, 4.0f, 4.0f};
    blink_softmax(uniform, 5);
    for (int i = 0; i < 5; ++i) {
        CHECK_NEAR(uniform[i], 0.2, 1e-6);
    }
}

static void test_matmul(void)
{
    /* out[r] = scale[r] * sum_c w[r][c] * x[c] */
    const int8_t weights[6] = {1, 2, 3, -1, 0, 4};
    const float scale[2] = {0.5f, 2.0f};
    const float x[3] = {1.0f, 10.0f, 100.0f};
    blink_tensor tensor = {weights, scale, 2u, 3u, BLINK_DTYPE_I8, NULL, NULL};
    float out[2];

    blink_matmul_i8(out, &tensor, x);
    CHECK_NEAR(out[0], 0.5 * (1 * 1 + 2 * 10 + 3 * 100), 1e-4);
    CHECK_NEAR(out[1], 2.0 * (-1 * 1 + 0 * 10 + 4 * 100), 1e-4);
}

static void test_dwconv(void)
{
    /* One channel, kernel 3, weights [10, 1, 100] over the signal [1, 2, 3].
     * Centred with zero padding: out[t] = sum_k w[k] * in[t + k - 1]. */
    const float weight[3] = {10.0f, 1.0f, 100.0f};
    const float in[3] = {1.0f, 2.0f, 3.0f};
    float out[3] = {0.0f, 0.0f, 0.0f};

    blink_dwconv_accum(out, in, weight, 3u, 1u, 3u);
    CHECK_NEAR(out[0], 1.0 * 1 + 100.0 * 2, 1e-4);
    CHECK_NEAR(out[1], 10.0 * 1 + 1.0 * 2 + 100.0 * 3, 1e-4);
    CHECK_NEAR(out[2], 10.0 * 2 + 1.0 * 3, 1e-4);

    /* accumulation, not assignment */
    blink_dwconv_accum(out, in, weight, 3u, 1u, 3u);
    CHECK_NEAR(out[0], 2.0 * (1.0 + 200.0), 1e-4);

    /* a length-one sequence sees only the centre tap */
    float single_out[1] = {0.0f};
    const float single_in[1] = {5.0f};
    blink_dwconv_accum(single_out, single_in, weight, 1u, 1u, 3u);
    CHECK_NEAR(single_out[0], 5.0, 1e-4);
}

/* The SIMD path must agree with the scalar definition. The two are not
 * bit-identical -- widening the loop reorders the fp32 additions -- so this
 * bounds the disagreement instead of demanding equality. */
static void test_matmul_simd_agrees_with_scalar(void)
{
    printf("simd path: %s\n", blink_backend());

    uint32_t seed = 0x5EED1234u;
    for (uint32_t cols = 1; cols <= 200; cols += 7) {
        int8_t weights[256];
        float x[256];
        for (uint32_t c = 0; c < cols; ++c) {
            seed = seed * 1664525u + 1013904223u;
            weights[c] = (int8_t)((int32_t)(seed >> 24) - 127);
            seed = seed * 1664525u + 1013904223u;
            x[c] = (float)((double)(seed >> 8) / 8388608.0 - 1.0);
        }
        const float scale = 0.0123f;
        blink_tensor tensor = {weights, &scale, 1u, cols, BLINK_DTYPE_I8,
                               NULL, NULL};
        float out = 0.0f;
        blink_matmul_i8(&out, &tensor, x);

        const double expected = (double)blink_dot_i8_scalar(weights, x, cols) *
                                (double)scale;
        /* Relative to the magnitude of the terms, not to the sum, because the
         * sum can cancel to near zero. */
        double magnitude = 0.0;
        for (uint32_t c = 0; c < cols; ++c) {
            magnitude += fabs((double)weights[c] * (double)x[c]);
        }
        CHECK_NEAR(out, expected, 1e-6 * magnitude * scale + 1e-9);
    }

    /* A zero row stays exactly zero on both paths. */
    int8_t zeros[64] = {0};
    float ones[64];
    for (int i = 0; i < 64; ++i) {
        ones[i] = 1.0f;
    }
    const float unit = 1.0f;
    blink_tensor zero_tensor = {zeros, &unit, 1u, 64u, BLINK_DTYPE_I8, NULL, NULL};
    float out = 1.0f;
    blink_matmul_i8(&out, &zero_tensor, ones);
    CHECK_EQ(out == 0.0f, 1);
}

/* The row-blocked product must equal the one-vector product exactly, for
 * every group size and for columns that do and do not fill a SIMD step: it
 * changes how often a weight is widened, never what is summed or in what
 * order. The runtime's bit-identity guarantees rest on this. */
static void test_matmul_rows_is_bit_identical(void)
{
    enum { ROWS = 7, MAXC = 70, MAXV = 11 };
    uint32_t seed = 0xB10C4u;
    int8_t weights[ROWS * MAXC];
    float scales[ROWS];
    float x[MAXV * MAXC];
    for (uint32_t i = 0; i < ROWS * MAXC; ++i) {
        seed = seed * 1664525u + 1013904223u;
        weights[i] = (int8_t)((int32_t)(seed >> 24) - 128);
    }
    for (uint32_t r = 0; r < ROWS; ++r) {
        scales[r] = 0.001f * (float)(r + 1u);
    }
    for (uint32_t i = 0; i < MAXV * MAXC; ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = (float)((double)(seed >> 8) / 8388608.0 - 1.0);
    }
    const uint32_t col_cases[] = {1u, 15u, 16u, 17u, 32u, 64u, 70u};
    for (size_t k = 0; k < sizeof col_cases / sizeof *col_cases; ++k) {
        const uint32_t cols = col_cases[k];
        const blink_tensor tensor = {weights, scales, ROWS, cols, BLINK_DTYPE_I8,
                                     NULL, NULL};
        for (uint32_t count = 0; count <= MAXV; ++count) {
            float blocked[MAXV * (ROWS + 1)];
            float single[MAXV * (ROWS + 1)];
            memset(blocked, 0, sizeof blocked);
            memset(single, 0, sizeof single);
            /* strides wider than the data, as the runtime uses them */
            blink_matmul_i8_rows(blocked, ROWS + 1u, &tensor, x, MAXC, count);
            for (uint32_t v = 0; v < count; ++v) {
                blink_matmul_i8(single + v * (ROWS + 1u), &tensor,
                                x + (size_t)v * MAXC);
            }
            CHECK_EQ(memcmp(blocked, single, sizeof blocked), 0);
        }
    }
}

#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
/* The Accelerate path against the scalar definition. The runtime only takes
 * it for tensors carrying a dense copy, which the fixture tensors above do
 * not, so this builds one the way blink_model.c does and checks every row
 * count the runtime can pass, up to a full group and past it, with strides
 * wider than the data. Agreement is to fp32 rounding, not bit for bit. */
static void test_matmul_accelerate_agrees_with_scalar(void)
{
    enum { ROWS = 24, MAXC = 72, MAXV = BLINK_GROUP_ROWS + 3 };
    static int8_t weights[ROWS * MAXC];
    static float dense[ROWS * MAXC];
    static float x[MAXV * MAXC];
    float scales[ROWS];
    uint32_t seed = 11u;
    for (uint32_t i = 0; i < ROWS * MAXC; ++i) {
        seed = seed * 1664525u + 1013904223u;
        weights[i] = (int8_t)((int32_t)(seed >> 24) - 128);
    }
    for (uint32_t r = 0; r < ROWS; ++r) {
        scales[r] = 0.002f * (float)(r + 1u);
    }
    for (uint32_t i = 0; i < MAXV * MAXC; ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = (float)((double)(seed >> 8) / 8388608.0 - 1.0);
    }
    const uint32_t col_cases[] = {1u, 16u, 17u, 64u, 72u};
    for (size_t k = 0; k < sizeof col_cases / sizeof *col_cases; ++k) {
        const uint32_t cols = col_cases[k];
        for (uint32_t r = 0; r < ROWS; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                dense[r * cols + c] = (float)weights[r * cols + c] * scales[r];
            }
        }
        const blink_tensor fast = {weights, scales, ROWS, cols, BLINK_DTYPE_I8,
                                   dense, NULL};
        const uint32_t counts[] = {1u, 2u, 4u, 5u, BLINK_GROUP_ROWS, MAXV};
        for (size_t n = 0; n < sizeof counts / sizeof *counts; ++n) {
            static float out[MAXV * (ROWS + 1)];
            blink_matmul_i8_rows(out, ROWS + 1u, &fast, x, MAXC, counts[n]);
            for (uint32_t v = 0; v < counts[n]; ++v) {
                for (uint32_t r = 0; r < ROWS; ++r) {
                    const double expected =
                        (double)blink_dot_i8_scalar(weights + r * cols,
                                                    x + (size_t)v * MAXC, cols) *
                        scales[r];
                    CHECK_NEAR(out[v * (ROWS + 1u) + r], expected, 1e-4);
                }
            }
        }
    }
    CHECK_EQ(blink_accelerate_enabled(), 1);
}
#endif

#if defined(BLINK_W8A8) && BLINK_W8A8
/* The activation quantizer against its definition written out here: the
 * NEON path must produce the same bytes and the same scale. */
static void test_w8a8_quantize(void)
{
    enum { MAXN = 77 };
    float x[MAXN];
    int8_t got[MAXN];
    uint32_t seed = 5u;
    for (uint32_t n = 0; n <= MAXN; ++n) {
        for (uint32_t i = 0; i < n; ++i) {
            seed = seed * 1664525u + 1013904223u;
            x[i] = (float)((double)(seed >> 8) / 8388608.0 - 1.0) * 3.0f;
        }
        if (n > 3u) {
            x[1] = 0.0f;
            x[2] = -x[0];
        }
        const float scale = blink_quantize_row(got, x, n);
        float maximum = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            if (fabsf(x[i]) > maximum) maximum = fabsf(x[i]);
        }
        if (maximum == 0.0f) {
            CHECK(scale == 0.0f);
            continue;
        }
        CHECK(scale == maximum / 127.0f);
        const float inverse = 127.0f / maximum;
        for (uint32_t i = 0; i < n; ++i) {
            long q = lrintf(x[i] * inverse);
            q = q > 127L ? 127L : (q < -127L ? -127L : q);
            CHECK_EQ((long)got[i], q);
        }
    }
    float zeros[9] = {0};
    int8_t out[9];
    CHECK(blink_quantize_row(out, zeros, 9u) == 0.0f);
    for (int i = 0; i < 9; ++i) CHECK_EQ(out[i], 0);
}

/* Scalar, SDOT and I8MM are the same integer sums, so their outputs must be
 * identical bytes, over odd and even rows, columns and group sizes. The
 * result must also match the definition computed in double. */
static void test_w8a8_kernels_agree(void)
{
    enum { ROWS = 13, MAXC = 72, MAXV = 9 };
    static int8_t weights[ROWS * MAXC];
    static float x[MAXV * MAXC];
    static int8_t scratch[2 * BLINK_ROW_BLOCK * MAXC];
    static int8_t packed[ROWS * MAXC];
    float scales[ROWS];
    uint32_t seed = 21u;
    for (uint32_t i = 0; i < ROWS * MAXC; ++i) {
        seed = seed * 1664525u + 1013904223u;
        weights[i] = (int8_t)((int32_t)(seed >> 24) - 128);
    }
    for (uint32_t r = 0; r < ROWS; ++r) scales[r] = 0.003f * (float)(r + 1u);
    for (uint32_t i = 0; i < MAXV * MAXC; ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = (float)((double)(seed >> 8) / 8388608.0 - 1.0);
    }

    const int original = blink_w8a8_kernel();
    int kernels[5], available = 0;
    for (int k = 0; k < 5; ++k) {
        if (blink_w8a8_force_kernel(k) == k) kernels[available++] = k;
    }
    const uint32_t col_cases[] = {1u, 7u, 8u, 9u, 16u, 17u, 31u, 32u, 64u, 72u};
    const uint32_t row_cases[] = {1u, 2u, 3u, 4u, 8u, ROWS};
    for (size_t ci = 0; ci < sizeof col_cases / sizeof *col_cases; ++ci) {
        for (size_t ri = 0; ri < sizeof row_cases / sizeof *row_cases; ++ri) {
            const uint32_t cols = col_cases[ci], rows = row_cases[ri];
            /* packed the way blink_model.c packs it, so the I8MM kernel runs
             * on its real layout rather than falling back */
            blink_pack_i8mm(packed, weights, rows, cols);
            const blink_tensor w = {weights, scales, rows, cols, BLINK_DTYPE_I8,
                                    NULL, packed};
            for (uint32_t count = 1; count <= MAXV; ++count) {
                static float reference[MAXV * (ROWS + 1)];
                static float out[MAXV * (ROWS + 1)];
                memset(reference, 0, sizeof reference);
                blink_w8a8_force_kernel(0);
                blink_matmul_w8a8_rows(reference, ROWS + 1u, &w, x, MAXC,
                                       count, scratch);
                for (int k = 1; k < available; ++k) {
                    blink_w8a8_force_kernel(kernels[k]);
                    memset(out, 0, sizeof out);
                    blink_matmul_w8a8_rows(out, ROWS + 1u, &w, x, MAXC, count,
                                           scratch);
                    CHECK_EQ(memcmp(out, reference, sizeof out), 0);
                }
                for (uint32_t v = 0; v < count; ++v) {
                    int8_t q[MAXC];
                    const float sx = blink_quantize_row(q, x + (size_t)v * MAXC,
                                                        cols);
                    for (uint32_t r = 0; r < rows; ++r) {
                        double sum = 0.0;
                        for (uint32_t c = 0; c < cols; ++c) {
                            sum += (double)weights[r * cols + c] * q[c];
                        }
                        CHECK_NEAR(reference[v * (ROWS + 1u) + r],
                                   sum * scales[r] * sx, 1e-5);
                    }
                }
            }
        }
    }
    blink_w8a8_force_kernel(original);
    printf("w8a8 kernels compared: %d (current %s)\n", available,
           blink_backend());
}
#endif

static void test_pool(void)
{
    /* Two channels, four positions, stride two: pairwise means. */
    float data[8] = {1.0f, 10.0f, 3.0f, 30.0f, 5.0f, 50.0f, 9.0f, 90.0f};
    CHECK_EQ(blink_pool(data, 4u, 2u, 2u), 2u);
    CHECK_NEAR(data[0], 2.0, 1e-6);
    CHECK_NEAR(data[1], 20.0, 1e-6);
    CHECK_NEAR(data[2], 7.0, 1e-6);
    CHECK_NEAR(data[3], 70.0, 1e-6);

    /* A short final window is divided by its real count, not by the stride. */
    float ragged[5] = {1.0f, 2.0f, 3.0f, 4.0f, 9.0f};
    CHECK_EQ(blink_pool(ragged, 5u, 1u, 2u), 3u);
    CHECK_NEAR(ragged[0], 1.5, 1e-6);
    CHECK_NEAR(ragged[1], 3.5, 1e-6);
    CHECK_NEAR(ragged[2], 9.0, 1e-6);

    /* Stride one and an empty sequence are identities. */
    float untouched[3] = {1.0f, 2.0f, 3.0f};
    CHECK_EQ(blink_pool(untouched, 3u, 1u, 1u), 3u);
    CHECK_NEAR(untouched[1], 2.0, 1e-9);
    CHECK_EQ(blink_pool(untouched, 0u, 1u, 4u), 0u);

    /* A stride larger than the sequence collapses it to one position. */
    float collapse[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    CHECK_EQ(blink_pool(collapse, 4u, 1u, 16u), 1u);
    CHECK_NEAR(collapse[0], 5.0, 1e-6);
}

static void test_dot(void)
{
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {0.5f, -1.0f, 2.0f, 0.0f};
    CHECK_NEAR(blink_dot(a, b, 4), 0.5 - 2.0 + 6.0, 1e-5);
    CHECK_NEAR(blink_dot(a, b, 0), 0.0, 1e-9);
}

static void test_bigram(void)
{
    /* Indices stay inside the table and the sentinel predecessor at position
     * zero is distinguishable from a real leading NUL byte only by the byte
     * itself, which is the documented behaviour. */
    for (uint32_t buckets = 8; buckets <= 4096; buckets *= 2) {
        for (int previous = 0; previous < 256; previous += 17) {
            for (int current = 0; current < 256; current += 13) {
                const uint32_t index = blink_bigram_index(
                    (uint8_t)previous, (uint8_t)current, buckets);
                CHECK(index < buckets);
            }
        }
    }
    /* order matters: the hash is not symmetric */
    CHECK(blink_bigram_index(1, 2, 4096) != blink_bigram_index(2, 1, 4096));

    /* the mixing spreads across the table rather than clustering */
    int seen[256];
    memset(seen, 0, sizeof seen);
    for (int previous = 0; previous < 256; ++previous) {
        for (int current = 0; current < 256; ++current) {
            seen[blink_bigram_index((uint8_t)previous, (uint8_t)current, 256)] = 1;
        }
    }
    int covered = 0;
    for (int i = 0; i < 256; ++i) {
        covered += seen[i];
    }
    CHECK_EQ(covered, 256);
}

static void test_crc32(void)
{
    /* Published IEEE CRC-32 vectors. */
    CHECK_EQ(blink_crc32(0u, "", 0), 0u);
    CHECK_EQ(blink_crc32(0u, "a", 1), 0xE8B7BE43u);
    CHECK_EQ(blink_crc32(0u, "123456789", 9), 0xCBF43926u);

    /* chaining a split buffer equals one pass over the whole buffer */
    const char *text = "the quick brown fox";
    const uint32_t whole = blink_crc32(0u, text, 19);
    const uint32_t split = blink_crc32(blink_crc32(0u, text, 7), text + 7, 12);
    CHECK_EQ(split, whole);
}

#ifdef __x86_64__
/* Every x86 level this CPU has, one after the other: each must agree with the
 * scalar definition, and within each the row-blocked product must stay
 * bit-identical to the one-vector product. */
static void test_x86_levels(void)
{
    const int original = blink_x86_level();
    int tested = 0;
    for (int level = 0; level <= 2; ++level) {
        if (blink_x86_force_level(level) != level) {
            continue;
        }
        ++tested;
        test_matmul_simd_agrees_with_scalar();
        test_matmul_rows_is_bit_identical();
    }
    blink_x86_force_level(original);
    printf("x86 levels tested: %d (current %s)\n", tested, blink_backend());
}
#endif

int main(void)
{
    test_layernorm();
    test_softmax();
    test_matmul();
    test_matmul_simd_agrees_with_scalar();
    test_matmul_rows_is_bit_identical();
#ifdef __x86_64__
    test_x86_levels();
#endif
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
    test_matmul_accelerate_agrees_with_scalar();
#endif
#if defined(BLINK_W8A8) && BLINK_W8A8
    test_w8a8_quantize();
    test_w8a8_kernels_agree();
#endif
    test_pool();
    test_dwconv();
    test_dot();
    test_bigram();
    test_crc32();
    TEST_MAIN_END();
}
