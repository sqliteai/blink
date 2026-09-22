/* Latency and throughput benchmark.
 *
 * Reports percentiles rather than a mean, because a mean hides the tail that
 * an embedded caller actually feels. Four paths are measured separately:
 *
 *   decide       state, menu and question every time (the cold path)
 *   cached       state and menu held, only the question changes
 *   cached_batch8  as cached, eight questions per blink_score_batch call
 *   menu         the state held, a new option list on every decision
 *
 * The split matters: the state encoder is the expensive part, and the point of
 * the split-encoder design is that it can be paid once.
 *
 * Output is one JSON object per configuration on stdout.
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* posix_memalign, clock_gettime and mmap are POSIX, not C99. glibc hides
 * them under -std=c99 unless a POSIX level is requested before any header;
 * Darwin exposes them anyway, and defining the macro there would hide other
 * interfaces (bench_memory's mach headers among them), so it is Linux only. */
#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "blink.h"

/* Where the synthesised fixtures live: build/, or build-accelerate/ for
 * `make ACCELERATE=1`, which sets it. */
#ifndef BLINK_BUILD_DIR
#define BLINK_BUILD_DIR "build"
#endif


/* Samples are capped by a wall-clock budget as well as a count, so the same
 * benchmark finishes in a predictable time whether it is timing a 390k-
 * parameter model or a 7.8M-parameter one. The achieved sample count is
 * reported with every row. */
#define WARMUP 50
#define MAX_SAMPLES 2000
#define MIN_SAMPLES 30
#define BUDGET_SECONDS 1.5

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare(const void *a, const void *b)
{
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double percentile(double *sorted, int count, double fraction)
{
    const int index = (int)(fraction * (count - 1));
    return sorted[index];
}

static const char *OPTIONS[16] = {
    "delivery and logistics", "billing and payments",
    "account access and sign-in", "device hardware repair",
    "privacy and data rights", "sales and product evaluation",
    "warranty replacement", "developer platform support",
    "enterprise onboarding", "fraud review",
    "returns processing", "network operations",
    "accessibility support", "legal and compliance",
    "partner integrations", "general enquiries",
};
static size_t LENGTHS[16];

/* A deterministic filler state of the requested byte length. */
static char *make_state(uint32_t length)
{
    static const char *words[] = {
        "order", "refund", "shipment", "invoice", "customer", "device",
        "network", "account", "ticket", "escalation", "warranty", "session",
    };
    char *buffer = (char *)malloc(length + 1u);
    uint32_t used = 0;
    uint32_t index = 0;
    while (used < length) {
        const char *word = words[index++ % 12u];
        const uint32_t size = (uint32_t)strlen(word);
        for (uint32_t i = 0; i < size && used < length; ++i) {
            buffer[used++] = word[i];
        }
        if (used < length) {
            buffer[used++] = ' ';
        }
    }
    buffer[length] = '\0';
    return buffer;
}

static void report(const char *model_name, const char *path, uint32_t state_bytes,
                   uint32_t options, double *samples, int count,
                   size_t arena_bytes)
{
    qsort(samples, (size_t)count, sizeof *samples, compare);
    double total = 0.0;
    for (int i = 0; i < count; ++i) {
        total += samples[i];
    }
    const double mean = total / count;
    printf(
        "{\"model\":\"%s\",\"path\":\"%s\",\"state_bytes\":%u,\"options\":%u,"
        "\"samples\":%d,\"mean_us\":%.3f,\"p50_us\":%.3f,\"p90_us\":%.3f,"
        "\"p99_us\":%.3f,\"max_us\":%.3f,\"decisions_per_second\":%.1f,"
        "\"arena_bytes\":%zu}\n",
        model_name, path, state_bytes, options, count, mean * 1e6,
        percentile(samples, count, 0.50) * 1e6,
        percentile(samples, count, 0.90) * 1e6,
        percentile(samples, count, 0.99) * 1e6, samples[count - 1] * 1e6,
        1.0 / mean, arena_bytes);
}

static void measure(blink_model *model, const char *name, uint32_t state_bytes,
                    uint32_t options)
{
    blink_limits limits = {0u, 0u, 16u, 0u};
    blink_status status = BLINK_OK;
    blink_session *session = blink_session_create(model, &limits, &status);
    if (!session) {
        fprintf(stderr, "session: %s\n", blink_status_string(status));
        exit(1);
    }
    const size_t arena = blink_session_size(model, &limits);

    char *state = make_state(state_bytes);
    const char *question = "Which team should handle this ticket?";
    const size_t question_length = strlen(question);
    float probabilities[16];
    double *samples = (double *)malloc(MAX_SAMPLES * sizeof *samples);
    int taken;

    /* cold path: everything re-encoded */
    for (int i = 0; i < WARMUP; ++i) {
        blink_decide(session, state, state_bytes, question, question_length,
                     OPTIONS, LENGTHS, options, probabilities, NULL);
    }
    taken = 0;
    for (double deadline = now_seconds() + BUDGET_SECONDS;
         taken < MAX_SAMPLES && (taken < MIN_SAMPLES || now_seconds() < deadline);
         ++taken) {
        const double start = now_seconds();
        blink_decide(session, state, state_bytes, question, question_length,
                     OPTIONS, LENGTHS, options, probabilities, NULL);
        samples[taken] = now_seconds() - start;
    }
    report(name, "decide", state_bytes, options, samples, taken, arena);

    /* warm path: the state and the menu are cached, only the question moves */
    blink_state_set(session, state, state_bytes);
    blink_menu_set(session, OPTIONS, LENGTHS, options);
    for (int i = 0; i < WARMUP; ++i) {
        blink_score(session, question, question_length, probabilities, NULL);
    }
    taken = 0;
    for (double deadline = now_seconds() + BUDGET_SECONDS;
         taken < MAX_SAMPLES && (taken < MIN_SAMPLES || now_seconds() < deadline);
         ++taken) {
        const double start = now_seconds();
        blink_score(session, question, question_length, probabilities, NULL);
        samples[taken] = now_seconds() - start;
    }
    report(name, "cached_state", state_bytes, options, samples, taken, arena);

    /* the same, with BATCH questions per blink_score_batch call; each sample
     * is one call divided by BATCH, so it compares with cached_state per
     * decision */
    {
        enum { BATCH = 8 };
        const char *questions[BATCH];
        size_t lengths[BATCH];
        float batch_probabilities[BATCH * 16];
        for (int i = 0; i < BATCH; ++i) {
            questions[i] = question;
            lengths[i] = question_length;
        }
        for (int i = 0; i < WARMUP; ++i) {
            blink_score_batch(session, questions, lengths, BATCH,
                              batch_probabilities, NULL);
        }
        taken = 0;
        for (double deadline = now_seconds() + BUDGET_SECONDS;
             taken < MAX_SAMPLES &&
             (taken < MIN_SAMPLES || now_seconds() < deadline);
             ++taken) {
            const double start = now_seconds();
            blink_score_batch(session, questions, lengths, BATCH,
                              batch_probabilities, NULL);
            samples[taken] = (now_seconds() - start) / (double)BATCH;
        }
        report(name, "cached_batch8", state_bytes, options, samples, taken,
               arena);
    }

    /* a new menu against a held state */
    taken = 0;
    for (double deadline = now_seconds() + BUDGET_SECONDS;
         taken < MAX_SAMPLES && (taken < MIN_SAMPLES || now_seconds() < deadline);
         ++taken) {
        const double start = now_seconds();
        blink_menu_set(session, OPTIONS, LENGTHS, options);
        blink_score(session, question, question_length, probabilities, NULL);
        samples[taken] = now_seconds() - start;
    }
    report(name, "new_menu", state_bytes, options, samples, taken, arena);

    free(samples);
    free(state);
    blink_session_free(session);
}

int main(int argc, char **argv)
{
    for (int i = 0; i < 16; ++i) {
        LENGTHS[i] = strlen(OPTIONS[i]);
    }

    const char *paths[2] = {BLINK_BUILD_DIR "/synth-tiny.blink",
                            BLINK_BUILD_DIR "/synth-small.blink"};
    const char *names[2] = {"blink-tiny", "blink-small"};
    int count = 2;
    if (argc > 1) {
        paths[0] = argv[1];
        names[0] = argv[1];
        count = 1;
    }

    for (int m = 0; m < count; ++m) {
        blink_status status = BLINK_OK;
        blink_model *model = blink_model_open_file(paths[m], 1, &status);
        if (!model) {
            fprintf(stderr, "skipping %s: %s\n", paths[m],
                    blink_status_string(status));
            continue;
        }
        blink_model_info info;
        blink_model_get_info(model, &info);

        const uint32_t sizes[3] = {64u, info.max_state / 2u, info.max_state};
        for (int s = 0; s < 3; ++s) {
            measure(model, info.name, sizes[s], 4u);
        }
        measure(model, info.name, info.max_state, 2u);
        measure(model, info.name, info.max_state, 16u);
        blink_model_close(model);
    }
    return 0;
}
