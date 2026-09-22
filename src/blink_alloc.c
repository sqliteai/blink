/* The only allocating helpers in the runtime.
 *
 * They live in their own translation unit so that tests/c/check_no_malloc.sh
 * can assert mechanically that the scoring path -- blink_runtime.o and
 * blink_kernels.o -- references no allocator symbol at all. Callers that
 * cannot tolerate malloc use blink_session_init with their own buffer and
 * never link against this file's behaviour on the hot path.
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* posix_memalign, clock_gettime and mmap are POSIX, not C99. glibc hides
 * them under -std=c99 unless a POSIX level is requested before any header;
 * Darwin exposes them anyway, and defining the macro there would hide other
 * interfaces (bench_memory's mach headers among them), so it is Linux only. */
#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdlib.h>

#include "blink_internal.h"

blink_session *blink_session_create(const blink_model *model,
                                    const blink_limits *limits,
                                    blink_status *status)
{
    if (!model) {
        if (status) *status = BLINK_E_INVALID;
        return NULL;
    }
    const size_t bytes = blink_session_size(model, limits);
    void *memory = NULL;
    if (posix_memalign(&memory, BLINK_ALIGN, bytes) != 0) {
        if (status) *status = BLINK_E_NOSPACE;
        return NULL;
    }
    blink_session *session =
        blink_session_init(memory, bytes, model, limits, status);
    if (!session) {
        free(memory);
        return NULL;
    }
    session->owned = memory;
    return session;
}

void blink_session_free(blink_session *session)
{
    if (session && session->owned) {
        free(session->owned);
    }
}
