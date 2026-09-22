/* Input edge cases: byte transparency, boundary lengths and option counts.
 * SPDX-License-Identifier: Apache-2.0 */

#include "blink_internal.h"
#include "harness.h"

static int finite_distribution(const float *values, uint32_t count)
{
    double total = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!(values[i] >= 0.0f) || !(values[i] <= 1.0f)) {
            return 0;
        }
        total += values[i];
    }
    return fabs(total - 1.0) < 1e-4;
}

/* Blink is byte-level: it must accept any byte, including NUL and invalid
 * UTF-8, because callers pass raw program state rather than curated text. */
static void test_byte_transparency(blink_session *session, uint32_t max_state)
{
    uint8_t all_bytes[256];
    for (int i = 0; i < 256; ++i) {
        all_bytes[i] = (uint8_t)i;
    }
    const uint32_t length = max_state < 256u ? max_state : 256u;
    CHECK_EQ(blink_state_set(session, (const char *)all_bytes, length), BLINK_OK);

    const char *options[3] = {"alpha", "beta", "gamma"};
    const size_t lengths[3] = {5u, 4u, 5u};
    CHECK_EQ(blink_menu_set(session, options, lengths, 3u), BLINK_OK);

    float probabilities[3];
    blink_result result;
    CHECK_EQ(blink_score(session, "?", 1u, probabilities, &result), BLINK_OK);
    CHECK(finite_distribution(probabilities, 3u));
    CHECK_EQ(result.state_bytes, length);

    /* embedded NUL bytes are data, not terminators */
    const char embedded[9] = {'a', 'b', 0, 'c', 'd', 0, 0, 'e', 'f'};
    CHECK_EQ(blink_state_set(session, embedded, 9u), BLINK_OK);
    CHECK_EQ(blink_score(session, "?", 1u, probabilities, &result), BLINK_OK);
    CHECK_EQ(result.state_bytes, 9u);
    CHECK(finite_distribution(probabilities, 3u));

    /* multi-byte UTF-8 and lone continuation bytes both go through */
    const char *utf8 = "\xc3\xa8 caff\xc3\xa8 \xe2\x9c\x93 \xf0\x9f\x9a\x80";
    CHECK_EQ(blink_state_set(session, utf8, strlen(utf8)), BLINK_OK);
    CHECK_EQ(blink_score(session, "?", 1u, probabilities, &result), BLINK_OK);
    CHECK(finite_distribution(probabilities, 3u));

    const char invalid[3] = {(char)0x80, (char)0xFE, (char)0xFF};
    CHECK_EQ(blink_state_set(session, invalid, 3u), BLINK_OK);
    CHECK_EQ(blink_score(session, "?", 1u, probabilities, &result), BLINK_OK);
    CHECK(finite_distribution(probabilities, 3u));
}

static void test_boundary_lengths(blink_session *session,
                                  const blink_model_info *info)
{
    const char *options[2] = {"yes", "no"};
    const size_t lengths[2] = {3u, 2u};
    float probabilities[2];
    blink_result result;
    CHECK_EQ(blink_menu_set(session, options, lengths, 2u), BLINK_OK);

    /* a single byte of state */
    CHECK_EQ(blink_state_set(session, "x", 1u), BLINK_OK);
    CHECK_EQ(blink_score(session, "q", 1u, probabilities, &result), BLINK_OK);
    CHECK(finite_distribution(probabilities, 2u));

    /* an empty question: the state alone carries the decision */
    CHECK_EQ(blink_score(session, NULL, 0u, probabilities, &result), BLINK_OK);
    CHECK_EQ(result.question_bytes, 0u);
    CHECK(finite_distribution(probabilities, 2u));

    /* an empty state with a question still works */
    CHECK_EQ(blink_state_set(session, NULL, 0u), BLINK_OK);
    CHECK_EQ(blink_score(session, "q", 1u, probabilities, &result), BLINK_OK);
    CHECK_EQ(result.state_bytes, 0u);
    CHECK(finite_distribution(probabilities, 2u));

    /* but an empty state and an empty question have nothing to read */
    CHECK_EQ(blink_score(session, NULL, 0u, probabilities, &result),
             BLINK_E_INVALID);

    /* exactly at the limits, and one byte past */
    char *buffer = (char *)malloc(info->max_state + 1u);
    memset(buffer, 'a', info->max_state + 1u);
    CHECK_EQ(blink_state_set(session, buffer, info->max_state), BLINK_OK);
    CHECK_EQ(blink_state_set(session, buffer, info->max_state + 1u), BLINK_E_LIMIT);
    CHECK_EQ(blink_score(session, buffer, info->max_question, probabilities,
                         &result),
             BLINK_OK);
    CHECK_EQ(result.state_bytes, info->max_state);
    CHECK_EQ(result.question_bytes, info->max_question);
    CHECK(finite_distribution(probabilities, 2u));
    CHECK_EQ(blink_score(session, buffer, info->max_question + 1u,
                         probabilities, &result),
             BLINK_E_LIMIT);

    /* an option exactly at the limit, and one byte past */
    const char *long_options[2];
    size_t long_lengths[2];
    long_options[0] = buffer;
    long_options[1] = "no";
    long_lengths[0] = info->max_option;
    long_lengths[1] = 2u;
    CHECK_EQ(blink_menu_set(session, long_options, long_lengths, 2u), BLINK_OK);
    long_lengths[0] = info->max_option + 1u;
    CHECK_EQ(blink_menu_set(session, long_options, long_lengths, 2u),
             BLINK_E_LIMIT);
    free(buffer);
}

/* Growing the menu must keep producing a normalised distribution, and the
 * relative order of two options must not depend on how many others are
 * present -- that is what makes the softmax a conditional score rather than an
 * arbitrary ranking. */
static void test_option_counts(const blink_model *model)
{
    static const char *pool[16] = {
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliett", "kilo", "lima", "mike", "november",
        "oscar", "papa",
    };
    size_t lengths[16];
    for (int i = 0; i < 16; ++i) {
        lengths[i] = strlen(pool[i]);
    }

    blink_limits limits = {0u, 0u, 16u, 0u};
    blink_status status = BLINK_OK;
    blink_session *session = blink_session_create(model, &limits, &status);
    CHECK_EQ(status, BLINK_OK);

    const char *state = "The report names alpha and delta as the candidates.";
    CHECK_EQ(blink_state_set(session, state, strlen(state)), BLINK_OK);

    float probabilities[16];
    double ratio = 0.0;
    for (uint32_t count = 2u; count <= 16u; ++count) {
        CHECK_EQ(blink_menu_set(session, pool, lengths, count), BLINK_OK);
        CHECK_EQ(blink_score(session, "Which one?", 10u, probabilities, NULL),
                 BLINK_OK);
        CHECK(finite_distribution(probabilities, count));
        const double current = (double)probabilities[0] / probabilities[1];
        if (count == 2u) {
            ratio = current;
        } else {
            CHECK_NEAR(current / ratio, 1.0, 1e-3);
        }
    }

    /* one more than the session allows is refused */
    CHECK_EQ(blink_menu_set(session, pool, lengths, 17u), BLINK_E_LIMIT);
    blink_session_free(session);
}

/* Reordering a menu must permute the probabilities and nothing else. */
static void test_order_equivariance(const blink_model *model)
{
    blink_status status = BLINK_OK;
    blink_session *session = blink_session_create(model, NULL, &status);
    const char *forward[3] = {"shipping", "billing", "access"};
    const char *reverse[3] = {"access", "billing", "shipping"};
    size_t lengths[3] = {8u, 7u, 6u};
    size_t reverse_lengths[3] = {6u, 7u, 8u};
    float a[3], b[3];

    const char *state = "The customer cannot sign in after a password reset.";
    CHECK_EQ(blink_state_set(session, state, strlen(state)), BLINK_OK);
    CHECK_EQ(blink_menu_set(session, forward, lengths, 3u), BLINK_OK);
    CHECK_EQ(blink_score(session, "Which queue?", 12u, a, NULL), BLINK_OK);
    CHECK_EQ(blink_menu_set(session, reverse, reverse_lengths, 3u), BLINK_OK);
    CHECK_EQ(blink_score(session, "Which queue?", 12u, b, NULL), BLINK_OK);

    for (int i = 0; i < 3; ++i) {
        CHECK_NEAR(a[i], b[2 - i], 1e-6);
    }
    blink_session_free(session);
}

int main(int argc, char **argv)
{
    const char *path = model_path(argc, argv, BLINK_BUILD_DIR "/synth-tiny.blink");
    blink_status status = BLINK_E_IO;
    blink_model *model = blink_model_open_file(path, 1, &status);
    if (!model) {
        fprintf(stderr, "cannot open %s: %s\n", path,
                blink_status_string(status));
        return 2;
    }
    blink_model_info info;
    blink_model_get_info(model, &info);

    blink_session *session = blink_session_create(model, NULL, &status);
    CHECK_EQ(status, BLINK_OK);
    test_byte_transparency(session, info.max_state);
    test_boundary_lengths(session, &info);
    blink_session_free(session);

    test_option_counts(model);
    test_order_equivariance(model);

    blink_model_close(model);
    TEST_MAIN_END();
}
