/* Embedding Blink with no heap at all.
 *
 * This is the shape the runtime was designed for: the weights are a mapped
 * file (or a byte array linked into the binary) and the session lives in a
 * static buffer whose size is known at compile time. Nothing here calls
 * malloc, so the same code works in a firmware image or under an allocator
 * you are not allowed to touch on the hot path.
 *
 *   cc -std=c99 -O2 -Iinclude examples/embed_static.c build/libblink.a -lm
 *   ./a.out artifacts/blink-tiny-synthetic-s7.blink
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <string.h>

#include "blink.h"

/* Declare what this program will actually pass in. Smaller limits mean a
 * smaller arena, and the runtime refuses anything larger rather than
 * truncating it behind your back. */
static const blink_limits LIMITS = {
    .max_state = 192,
    .max_question = 64,
    .max_options = 4,
    .max_option = 32,
};

/* Sized by hand once, then checked at startup. blink_session_size is a pure
 * function of the model and the limits, so this check either passes always or
 * fails always -- it cannot pass in testing and fail in the field.
 *
 * The union forces the strictest fundamental alignment the ABI has, which
 * covers the runtime's 16-byte requirement without needing C11's _Alignas. */
static union {
    unsigned char bytes[192 * 1024];
    long double alignment;
} ARENA;

static const char *const OPTIONS[] = {
    "delivery and logistics",
    "billing and payments",
    "account access and sign-in",
};
static const size_t OPTION_COUNT = sizeof OPTIONS / sizeof OPTIONS[0];

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "artifacts/blink-tiny-synthetic-s7.blink";

    blink_status status = BLINK_OK;
    blink_model *model = blink_model_open_file(path, 1, &status);
    if (!model) {
        fprintf(stderr, "cannot open %s: %s\n", path,
                blink_status_string(status));
        return 1;
    }

    const size_t needed = blink_session_size(model, &LIMITS);
    if (needed > sizeof ARENA.bytes) {
        fprintf(stderr, "arena too small: need %zu bytes, have %zu\n", needed,
                sizeof ARENA.bytes);
        blink_model_close(model);
        return 1;
    }

    blink_session *session =
        blink_session_init(ARENA.bytes, sizeof ARENA.bytes, model, &LIMITS,
                           &status);
    if (!session) {
        fprintf(stderr, "session: %s\n", blink_status_string(status));
        blink_model_close(model);
        return 1;
    }
    printf("arena: %zu of %zu bytes used\n", needed, sizeof ARENA.bytes);

    size_t lengths[3];
    for (size_t i = 0; i < OPTION_COUNT; ++i) {
        lengths[i] = strlen(OPTIONS[i]);
    }
    if (blink_menu_set(session, OPTIONS, lengths, (uint32_t)OPTION_COUNT) !=
        BLINK_OK) {
        fprintf(stderr, "menu rejected\n");
        blink_model_close(model);
        return 1;
    }

    /* Two tickets, each asked one question. In a real service the state would
     * change per request and the menu would be set once; here both are shown. */
    static const char *const TICKETS[] = {
        "The parcel left the depot on Monday and has not arrived since.",
        "The customer cannot sign in after a password reset this morning.",
    };
    const char *question = "Which team should handle this ticket?";

    for (size_t t = 0; t < sizeof TICKETS / sizeof TICKETS[0]; ++t) {
        if (blink_state_set(session, TICKETS[t], strlen(TICKETS[t])) != BLINK_OK) {
            fprintf(stderr, "state rejected (over %u bytes?)\n", LIMITS.max_state);
            continue;
        }
        float probabilities[3];
        blink_result result;
        if (blink_score(session, question, strlen(question), probabilities,
                        &result) != BLINK_OK) {
            fprintf(stderr, "score failed\n");
            continue;
        }
        printf("\n%s\n", TICKETS[t]);
        for (size_t i = 0; i < OPTION_COUNT; ++i) {
            printf("  %-28s %.4f%s\n", OPTIONS[i], (double)probabilities[i],
                   i == result.argmax ? "  <-" : "");
        }
        printf("  confidence %.4f, margin %.4f, %.1f us\n",
               (double)result.confidence, (double)result.margin,
               (result.encode_seconds + result.head_seconds) * 1e6);
    }

    /* No free for the session: it never owned anything. */
    blink_model_close(model);
    return 0;
}
