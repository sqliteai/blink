/* Minimal assertion harness for the C tests. SPDX-License-Identifier: Apache-2.0 */
#ifndef BLINK_TEST_HARNESS_H
#define BLINK_TEST_HARNESS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int blink_test_failures = 0;
static int blink_test_checks = 0;

#define CHECK(condition)                                                       \
    do {                                                                       \
        ++blink_test_checks;                                                   \
        if (!(condition)) {                                                    \
            ++blink_test_failures;                                             \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,   \
                    #condition);                                               \
        }                                                                      \
    } while (0)

#define CHECK_EQ(actual, expected)                                             \
    do {                                                                       \
        ++blink_test_checks;                                                   \
        const long long a_ = (long long)(actual);                              \
        const long long e_ = (long long)(expected);                            \
        if (a_ != e_) {                                                        \
            ++blink_test_failures;                                             \
            fprintf(stderr, "%s:%d: %s == %lld, expected %lld\n", __FILE__,    \
                    __LINE__, #actual, a_, e_);                                \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                \
    do {                                                                       \
        ++blink_test_checks;                                                   \
        const double a_ = (double)(actual);                                    \
        const double e_ = (double)(expected);                                  \
        if (!(fabs(a_ - e_) <= (tolerance))) {                                 \
            ++blink_test_failures;                                             \
            fprintf(stderr, "%s:%d: %s == %.9g, expected %.9g +- %.3g\n",      \
                    __FILE__, __LINE__, #actual, a_, e_, (double)(tolerance)); \
        }                                                                      \
    } while (0)

#define TEST_MAIN_END()                                                        \
    do {                                                                       \
        printf("%d checks, %d failures\n", blink_test_checks,                  \
               blink_test_failures);                                           \
        return blink_test_failures == 0 ? 0 : 1;                               \
    } while (0)

/* Every test takes an optional model path so the suite can run against a
 * trained container as well as the synthesised fixture. */
__attribute__((unused))
/* Where the synthesised fixtures live: build/, or build-accelerate/ for
 * `make ACCELERATE=1`, which sets it. */
#ifndef BLINK_BUILD_DIR
#define BLINK_BUILD_DIR "build"
#endif

static const char *model_path(int argc, char **argv, const char *fallback)
{
    return argc > 1 ? argv[1] : fallback;
}

#endif /* BLINK_TEST_HARNESS_H */
