/* Session arena behaviour, the public contract, and the caching invariant.
 * SPDX-License-Identifier: Apache-2.0 */

#include "blink_internal.h"
#include "harness.h"

/* A batch against the same questions scored one at a time. The default
 * build encodes both the same way term for term, so they must agree bit for
 * bit. The Accelerate build hands cblas_sgemm a different number of rows in
 * the two cases, and a row may round differently depending on its
 * neighbours, so there the check is agreement to fp32 rounding. */
static int same_scores(const float *a, const float *b, size_t n)
{
#if defined(BLINK_ACCELERATE) && BLINK_ACCELERATE
    for (size_t i = 0; i < n; ++i) {
        const float d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
        if (d > 1e-5f) return 0;
    }
    return 1;
#else
    return memcmp(a, b, n * sizeof(float)) == 0;
#endif
}

static const char *OPTIONS[4] = {
    "delivery and logistics", "billing and payments",
    "account access and sign-in", "device hardware repair",
};
static size_t LENGTHS[4];

static void fill_lengths(void)
{
    for (int i = 0; i < 4; ++i) {
        LENGTHS[i] = strlen(OPTIONS[i]);
    }
}

static void test_sizing(const blink_model *model)
{
    /* sizing is a pure function of the model and the limits */
    const size_t defaults = blink_session_size(model, NULL);
    CHECK(defaults > sizeof(blink_session));
    CHECK_EQ(blink_session_size(model, NULL), defaults);
    CHECK_EQ(defaults % BLINK_ALIGN, 0u);

    /* smaller limits mean a strictly smaller arena */
    blink_limits tight = {16u, 8u, 2u, 8u};
    const size_t small = blink_session_size(model, &tight);
    CHECK(small < defaults);

    blink_limits wide = {0u, 0u, 64u, 0u};
    CHECK(blink_session_size(model, &wide) > defaults);

    /* limits above the model's own capacity are clamped, not honoured */
    blink_limits absurd = {100000u, 100000u, 100000u, 100000u};
    blink_limits capped = {0u, 0u, 64u, 0u};
    CHECK_EQ(blink_session_size(model, &absurd),
             blink_session_size(model, &capped));

    CHECK_EQ(blink_session_size(NULL, NULL), 0u);
}

static void test_placement(const blink_model *model)
{
    blink_limits limits = {0u, 0u, 8u, 0u};
    const size_t bytes = blink_session_size(model, &limits);
    blink_status status = BLINK_E_IO;

    /* one byte short is refused rather than overrun */
    void *buffer = malloc(bytes + BLINK_ALIGN);
    uint8_t *aligned = (uint8_t *)(((uintptr_t)buffer + 63u) & ~(uintptr_t)63u);
    CHECK(blink_session_init(aligned, bytes - 1u, model, &limits, &status) == NULL);
    CHECK_EQ(status, BLINK_E_NOSPACE);

    /* exactly enough is accepted */
    blink_session *session =
        blink_session_init(aligned, bytes, model, &limits, &status);
    CHECK_EQ(status, BLINK_OK);
    CHECK(session != NULL);
    CHECK_EQ((void *)session, (void *)aligned);

    /* a misaligned pointer is refused */
    CHECK(blink_session_init(aligned + 1, bytes, model, &limits, &status) == NULL);
    CHECK_EQ(status, BLINK_E_INVALID);
    CHECK(blink_session_init(NULL, bytes, model, &limits, &status) == NULL);
    CHECK_EQ(status, BLINK_E_INVALID);

    /* a placed session owns nothing, so freeing it is a no-op */
    blink_session_free(session);
    free(buffer);
}

static void test_errors(const blink_model *model, blink_session *session)
{
    blink_model_info info;
    blink_model_get_info(model, &info);
    float probabilities[8];

    /* scoring before a state or a menu is an error, not undefined behaviour */
    CHECK_EQ(blink_score(session, "q", 1, probabilities, NULL), BLINK_E_STATE);

    CHECK_EQ(blink_state_set(session, "hello", 5), BLINK_OK);
    CHECK_EQ(blink_score(session, "q", 1, probabilities, NULL), BLINK_E_STATE);

    /* input longer than the session limit is refused, never truncated */
    char *oversize = (char *)malloc(info.max_state + 2u);
    memset(oversize, 'x', info.max_state + 1u);
    CHECK_EQ(blink_state_set(session, oversize, info.max_state + 1u), BLINK_E_LIMIT);
    /* and the previously cached state survives the rejection */
    CHECK_EQ(blink_state_set(session, "hello", 5), BLINK_OK);
    free(oversize);

    /* a menu needs at least two options and no more than the session allows */
    CHECK_EQ(blink_menu_set(session, OPTIONS, LENGTHS, 1u), BLINK_E_LIMIT);
    CHECK_EQ(blink_menu_set(session, OPTIONS, LENGTHS, 1000u), BLINK_E_LIMIT);

    /* an empty option is an error */
    const char *with_empty[2] = {"fine", ""};
    const size_t empty_lengths[2] = {4u, 0u};
    CHECK_EQ(blink_menu_set(session, with_empty, empty_lengths, 2u), BLINK_E_INVALID);

    /* null arguments */
    CHECK_EQ(blink_state_set(NULL, "x", 1), BLINK_E_INVALID);
    CHECK_EQ(blink_state_set(session, NULL, 4), BLINK_E_INVALID);
    CHECK_EQ(blink_menu_set(session, NULL, LENGTHS, 2u), BLINK_E_INVALID);
    CHECK_EQ(blink_score(session, "q", 1, NULL, NULL), BLINK_E_INVALID);
    CHECK(blink_last_logits(NULL, NULL) == NULL);
}

static void test_scoring(const blink_model *model, blink_session *session)
{
    blink_model_info info;
    blink_model_get_info(model, &info);
    const char *state = "The parcel left the depot on Monday and is still in transit.";
    const char *question = "Which team should handle this ticket?";
    float probabilities[8];
    blink_result result;

    CHECK_EQ(blink_decide(session, state, strlen(state), question,
                          strlen(question), OPTIONS, LENGTHS, 4u,
                          probabilities, &result),
             BLINK_OK);

    /* a proper distribution over exactly the declared options */
    double total = 0.0;
    for (uint32_t i = 0; i < 4u; ++i) {
        CHECK(probabilities[i] > 0.0f && probabilities[i] < 1.0f);
        total += probabilities[i];
    }
    CHECK_NEAR(total, 1.0, 1e-5);
    CHECK_EQ(result.options, 4u);
    CHECK(result.argmax < 4u);
    CHECK_NEAR(result.confidence, probabilities[result.argmax], 1e-7);
    CHECK(result.margin >= 0.0f);
    CHECK(result.entropy >= 0.0f);
    CHECK(result.entropy <= (float)log(4.0) + 1e-5f);
    CHECK_EQ(result.state_bytes, (uint32_t)strlen(state));
    CHECK_EQ(result.question_bytes, (uint32_t)strlen(question));
    CHECK(result.encode_seconds >= 0.0);
    CHECK(result.head_seconds >= 0.0);

    /* logits are exposed and consistent with the probabilities: a larger
     * logit must map to a larger probability */
    uint32_t count = 0;
    const float *logits = blink_last_logits(session, &count);
    CHECK_EQ(count, 4u);
    CHECK(logits != NULL);
    uint32_t best = 0;
    for (uint32_t i = 1; i < 4u; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    CHECK_EQ(best, result.argmax);

    /* attention is a distribution over the context positions */
    float attention[1024];
    const uint32_t written = blink_last_attention(session, 0u, attention, 1024u);
    CHECK_EQ(written, result.context_positions);
    double attention_total = 0.0;
    for (uint32_t i = 0; i < written; ++i) {
        CHECK(attention[i] >= 0.0f);
        attention_total += attention[i];
    }
    CHECK_NEAR(attention_total, 1.0, 1e-4);
    CHECK_EQ(blink_last_attention(session, 99u, attention, 1024u), 0u);

    /* a truncated capacity writes a prefix rather than overflowing */
    CHECK_EQ(blink_last_attention(session, 0u, attention, 4u), 4u);
}

static void test_cache_equivalence(blink_session *session)
{
    /* Setting the state once and asking many questions must give exactly the
     * same numbers as re-encoding the state for every question. This is the
     * whole point of the split encoder, so it is checked bit for bit. */
    const char *state =
        "Order 4471 was refunded on 3 March. The customer then asked why the "
        "invoice still shows an outstanding balance of 42 euro.";
    const char *questions[3] = {
        "Which team should handle this ticket?",
        "Was the refund issued?",
        "",
    };
    float fresh[4][8];
    float cached[4][8];

    for (int q = 0; q < 3; ++q) {
        CHECK_EQ(blink_decide(session, state, strlen(state), questions[q],
                              strlen(questions[q]), OPTIONS, LENGTHS, 4u,
                              fresh[q], NULL),
                 BLINK_OK);
    }

    CHECK_EQ(blink_state_set(session, state, strlen(state)), BLINK_OK);
    CHECK_EQ(blink_menu_set(session, OPTIONS, LENGTHS, 4u), BLINK_OK);
    for (int q = 0; q < 3; ++q) {
        CHECK_EQ(blink_score(session, questions[q], strlen(questions[q]),
                             cached[q], NULL),
                 BLINK_OK);
        for (uint32_t i = 0; i < 4u; ++i) {
            ++blink_test_checks;
            if (fresh[q][i] != cached[q][i]) {
                ++blink_test_failures;
                fprintf(stderr,
                        "cache changed question %d option %u: %.9g vs %.9g\n",
                        q, i, (double)fresh[q][i], (double)cached[q][i]);
            }
        }
    }

    /* Different questions over one state must be able to disagree, otherwise
     * the question is being ignored. */
    int differs = 0;
    for (uint32_t i = 0; i < 4u; ++i) {
        if (fresh[0][i] != fresh[1][i]) {
            differs = 1;
        }
    }
    CHECK(differs);
}

/* Scoring a batch of questions must give exactly what scoring them one at a
 * time gives: the batch only packs the questions side by side in the encoder.
 * Checked bit for bit, over enough questions that a tight session has to split
 * the batch into several passes, and with an empty question in the middle. */
static void test_batch_equivalence(const blink_model *model)
{
    const char *state =
        "Order 4471 was refunded on 3 March. The customer then asked why the "
        "invoice still shows an outstanding balance of 42 euro.";
    const char *pool[5] = {
        "Which team should handle this ticket?", "Was the refund issued?", "",
        "?", "Does the invoice contradict the refund that was issued in March?",
    };
    enum { COUNT = 23 };
    const char *questions[COUNT];
    size_t lengths[COUNT];
    for (int q = 0; q < COUNT; ++q) {
        questions[q] = pool[q % 5];
        lengths[q] = strlen(pool[q % 5]);
    }

    const blink_limits roomy = {0u, 0u, 8u, 0u};
    const blink_limits tight = {96u, 72u, 8u, 0u};
    const blink_limits *cases[2] = {&roomy, &tight};
    for (int c = 0; c < 2; ++c) {
        blink_status status;
        blink_session *one = blink_session_create(model, cases[c], &status);
        blink_session *many = blink_session_create(model, cases[c], &status);
        CHECK(one && many);
        CHECK_EQ(blink_state_set(one, state, 90u), BLINK_OK);
        CHECK_EQ(blink_state_set(many, state, 90u), BLINK_OK);
        CHECK_EQ(blink_menu_set(one, OPTIONS, LENGTHS, 4u), BLINK_OK);
        CHECK_EQ(blink_menu_set(many, OPTIONS, LENGTHS, 4u), BLINK_OK);

        float batched[COUNT * 4];
        blink_result results[COUNT];
        CHECK_EQ(blink_score_batch(many, questions, lengths, COUNT, batched,
                                   results),
                 BLINK_OK);
        for (int q = 0; q < COUNT; ++q) {
            float alone[4];
            blink_result result;
            CHECK_EQ(blink_score(one, questions[q], lengths[q], alone, &result),
                     BLINK_OK);
            CHECK(same_scores(alone, batched + q * 4, 4u));
            CHECK_EQ(result.argmax, results[q].argmax);
            CHECK_EQ(result.context_positions, results[q].context_positions);
        }

        /* the "last score" accessors describe the batch's last question */
        uint32_t n_one = 0, n_many = 0;
        const float *z_one = blink_last_logits(one, &n_one);
        const float *z_many = blink_last_logits(many, &n_many);
        CHECK(z_one && z_many && n_one == n_many);
        CHECK(same_scores(z_one, z_many, n_one));
        float a_one[128], a_many[128];
        const uint32_t k_one = blink_last_attention(one, 1u, a_one, 128u);
        const uint32_t k_many = blink_last_attention(many, 1u, a_many, 128u);
        CHECK_EQ(k_one, k_many);
        CHECK(same_scores(a_one, a_many, k_one));

        /* a batch with one bad question is refused before anything runs */
        char too_long[200];
        memset(too_long, 'q', sizeof too_long);
        const char *bad[2] = {pool[0], too_long};
        const size_t bad_lengths[2] = {lengths[0], sizeof too_long};
        CHECK_EQ(blink_score_batch(many, bad, bad_lengths, 2u, batched, NULL),
                 BLINK_E_LIMIT);
        z_many = blink_last_logits(many, &n_many);
        CHECK(same_scores(z_one, z_many, n_one));
        CHECK_EQ(blink_score_batch(many, questions, lengths, 0u, batched, NULL),
                 BLINK_E_INVALID);
        CHECK_EQ(blink_score_batch(many, NULL, lengths, 1u, batched, NULL),
                 BLINK_E_INVALID);

        blink_session_free(one);
        blink_session_free(many);
    }
}

static void test_determinism(const blink_model *model)
{
    const char *state = "Deterministic runs must agree exactly.";
    const char *question = "Which team?";
    float first[8], second[8];
    blink_status status;

    blink_session *a = blink_session_create(model, NULL, &status);
    CHECK_EQ(status, BLINK_OK);
    CHECK_EQ(blink_decide(a, state, strlen(state), question, strlen(question),
                          OPTIONS, LENGTHS, 3u, first, NULL),
             BLINK_OK);

    /* repeating in the same session */
    CHECK_EQ(blink_decide(a, state, strlen(state), question, strlen(question),
                          OPTIONS, LENGTHS, 3u, second, NULL),
             BLINK_OK);
    CHECK_EQ(memcmp(first, second, 3u * sizeof(float)), 0);

    /* and in a brand-new session over the same model */
    blink_session *b = blink_session_create(model, NULL, &status);
    CHECK_EQ(blink_decide(b, state, strlen(state), question, strlen(question),
                          OPTIONS, LENGTHS, 3u, second, NULL),
             BLINK_OK);
    CHECK_EQ(memcmp(first, second, 3u * sizeof(float)), 0);

    /* the option count is runtime-defined: two options and four options are
     * both valid against the same cached state */
    float two[2], four[4];
    CHECK_EQ(blink_state_set(a, state, strlen(state)), BLINK_OK);
    CHECK_EQ(blink_menu_set(a, OPTIONS, LENGTHS, 2u), BLINK_OK);
    CHECK_EQ(blink_score(a, question, strlen(question), two, NULL), BLINK_OK);
    CHECK_EQ(blink_menu_set(a, OPTIONS, LENGTHS, 4u), BLINK_OK);
    CHECK_EQ(blink_score(a, question, strlen(question), four, NULL), BLINK_OK);
    CHECK_NEAR((double)two[0] + two[1], 1.0, 1e-5);
    CHECK_NEAR((double)four[0] + four[1] + four[2] + four[3], 1.0, 1e-5);
    /* renormalising a subset must change the values, not merely reuse them */
    CHECK(two[0] != four[0]);

    blink_session_free(a);
    blink_session_free(b);
}

int main(int argc, char **argv)
{
    fill_lengths();
    const char *path = model_path(argc, argv, BLINK_BUILD_DIR "/synth-tiny.blink");
    blink_status status = BLINK_E_IO;
    blink_model *model = blink_model_open_file(path, 1, &status);
    if (!model) {
        fprintf(stderr, "cannot open %s: %s\n", path,
                blink_status_string(status));
        return 2;
    }

    test_sizing(model);
    test_placement(model);

    blink_session *session = blink_session_create(model, NULL, &status);
    CHECK_EQ(status, BLINK_OK);
    test_errors(model, session);
    test_scoring(model, session);
    test_cache_equivalence(session);
    blink_session_free(session);

    test_determinism(model);
    test_batch_equivalence(model);

    blink_model_close(model);
    TEST_MAIN_END();
}
