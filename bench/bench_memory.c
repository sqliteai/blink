/* Memory benchmark.
 *
 * Three numbers matter for an embedded caller and they are measured
 * separately, because they live in different places:
 *
 *   weights        bytes in the container, mapped read-only and never copied,
 *                  so they are shared by every process and every session
 *   arena          bytes of the one working buffer a session needs, which the
 *                  caller may place wherever it likes
 *   resident size  what the operating system actually charges, measured as
 *                  the growth from adding 64 concurrent sessions over one
 *                  shared model
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
#include <sys/resource.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "blink.h"

/* Where the synthesised fixtures live: build/, or build-accelerate/ for
 * `make ACCELERATE=1`, which sets it. */
#ifndef BLINK_BUILD_DIR
#define BLINK_BUILD_DIR "build"
#endif


static long peak_rss_kib(void)
{
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return usage.ru_maxrss / 1024; /* Darwin reports bytes */
#else
    return usage.ru_maxrss;        /* Linux reports kibibytes */
#endif
}

/* Current resident size, not the peak. The peak is a high-water mark for the
 * whole process, so it cannot show that freeing one thing made room for
 * another -- which is exactly what the sharing measurement below needs. */
static long resident_kib(void)
{
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (long)(info.resident_size / 1024u);
    }
    return -1;
#elif defined(__linux__)
    long pages = 0;
    FILE *file = fopen("/proc/self/statm", "r");
    if (file && fscanf(file, "%*s %ld", &pages) == 1) {
        fclose(file);
        return pages * (sysconf(_SC_PAGESIZE) / 1024);
    }
    if (file) {
        fclose(file);
    }
    return -1;
#else
    return -1;
#endif
}

static const char *OPTIONS[4] = {
    "delivery and logistics", "billing and payments",
    "account access and sign-in", "device hardware repair",
};
static size_t LENGTHS[4];

static void report_model(const char *path)
{
    blink_status status = BLINK_OK;
    blink_model *model = blink_model_open_file(path, 1, &status);
    if (!model) {
        fprintf(stderr, "skipping %s: %s\n", path, blink_status_string(status));
        return;
    }
    blink_model_info info;
    blink_model_get_info(model, &info);

    /* Arena size across a range of declared limits. Smaller limits buy a
     * smaller buffer with no change to the weights. */
    struct { const char *label; blink_limits limits; } cases[] = {
        {"minimum", {16u, 8u, 2u, 8u}},
        {"embedded", {128u, 64u, 4u, 32u}},
        {"default", {0u, 0u, 16u, 0u}},
        {"widest", {0u, 0u, 64u, 0u}},
    };

    char buffer[512];
    int written = snprintf(buffer, sizeof buffer,
        "{\"model\":\"%s\",\"path\":\"%s\",\"parameters\":%llu,"
        "\"weights_bytes\":%llu,\"bytes_per_parameter\":%.3f,\"arenas\":{",
        info.name, path, (unsigned long long)info.parameters,
        (unsigned long long)info.weights_bytes,
        (double)info.weights_bytes / (double)info.parameters);

    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        const size_t bytes = blink_session_size(model, &cases[i].limits);
        written += snprintf(buffer + written, sizeof buffer - (size_t)written,
                            "%s\"%s\":%zu", i ? "," : "", cases[i].label, bytes);
    }
    snprintf(buffer + written, sizeof buffer - (size_t)written, "}}");
    printf("%s\n", buffer);

    /* Exercise one session so the report is not made on an idle process, but
     * do not try to attribute an RSS delta to it: the process-wide figure is
     * dominated by whatever ran before. report_sharing is the measurement that
     * isolates the marginal cost. */
    blink_session *session = blink_session_create(model, NULL, &status);
    float probabilities[4];
    char state[512];
    memset(state, 'a', sizeof state);
    for (int i = 0; i < 200; ++i) {
        blink_decide(session, state,
                     info.max_state < sizeof state ? info.max_state
                                                   : sizeof state,
                     "Which team?", 11u, OPTIONS, LENGTHS, 4u, probabilities,
                     NULL);
    }
    blink_session_free(session);
    blink_model_close(model);
}

/* Many sessions over one model. The weights are mapped once, so the marginal
 * cost of a concurrent decision is one arena and nothing else. Every session
 * scores a full-length state so its arena is genuinely resident rather than
 * merely reserved. */
static void report_sharing(const char *path)
{
    blink_status status = BLINK_OK;
    blink_model *model = blink_model_open_file(path, 1, &status);
    if (!model) {
        return;
    }
    blink_model_info info;
    blink_model_get_info(model, &info);
    blink_limits limits = {0u, 0u, 4u, 0u};
    const size_t arena = blink_session_size(model, &limits);

    enum { SESSIONS = 64 };
    blink_session *sessions[SESSIONS];
    char *state = (char *)malloc(info.max_state);
    memset(state, 'a', info.max_state);
    float probabilities[4];

    const long before = resident_kib();
    for (int i = 0; i < SESSIONS; ++i) {
        sessions[i] = blink_session_create(model, &limits, &status);
        blink_decide(sessions[i], state, info.max_state, "Which team?", 11u,
                     OPTIONS, LENGTHS, 4u, probabilities, NULL);
    }
    const long with_sessions = resident_kib();
    for (int i = 0; i < SESSIONS; ++i) {
        blink_session_free(sessions[i]);
    }
    const long after_free = resident_kib();

    printf("{\"model\":\"%s\",\"sessions\":%d,\"arena_bytes_each\":%zu,"
           "\"arena_bytes_total_kib\":%zu,\"rss_kib_before\":%ld,"
           "\"rss_kib_with_sessions\":%ld,\"rss_kib_after_free\":%ld,"
           "\"rss_growth_kib\":%ld,\"weights_bytes_shared\":%llu,"
           "\"note\":\"growth should track the arenas, not the weights\"}\n",
           info.name, SESSIONS, arena, arena * SESSIONS / 1024u, before,
           with_sessions, after_free, with_sessions - before,
           (unsigned long long)info.weights_bytes);

    free(state);
    blink_model_close(model);
}

int main(int argc, char **argv)
{
    for (int i = 0; i < 4; ++i) {
        LENGTHS[i] = strlen(OPTIONS[i]);
    }
    /* The sharing measurement runs first, on a clean process, so that its
     * numbers are not polluted by mappings the other reports have already
     * created and released. */
    if (argc > 1) {
        report_sharing(argv[1]);
        for (int i = 1; i < argc; ++i) {
            report_model(argv[i]);
        }
        printf("{\"process_peak_rss_kib\":%ld,\"process_resident_kib\":%ld}\n",
               peak_rss_kib(), resident_kib());
        return 0;
    }
    report_sharing(BLINK_BUILD_DIR "/synth-tiny.blink");
    report_model(BLINK_BUILD_DIR "/synth-tiny.blink");
    report_model(BLINK_BUILD_DIR "/synth-small.blink");
    printf("{\"process_peak_rss_kib\":%ld,\"process_resident_kib\":%ld}\n",
           peak_rss_kib(), resident_kib());
    return 0;
}
