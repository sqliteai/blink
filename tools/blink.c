/* blink -- score one decision from the command line.
 *
 *   blink MODEL.blink --state TEXT [--question TEXT] --option TEXT --option TEXT
 *   blink MODEL.blink --info
 *   blink --version | --help
 *
 * Documented in docs/blink.1 (installed as the blink(1) man page) and
 * docs/CLI.md; tests/python/test_harness.py checks that every option in
 * usage() appears in both.
 *
 * Output is one JSON object. For batches, use eval/evaluate.py, which drives
 * the same library through the ctypes binding and reuses a cached state across
 * consecutive rows that share one.
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blink.h"

#define MAX_OPTIONS 32

static void usage(FILE *stream, const char *program)
{
    fprintf(stream,
            "usage: %s MODEL.blink --state TEXT [--question TEXT]"
            " --option TEXT --option TEXT [...]\n"
            "       %s MODEL.blink --info\n"
            "       %s --version | --help\n\n"
            "  --state TEXT      the unstructured program state\n"
            "  --question TEXT   the criterion, defined at call time\n"
            "  --option TEXT     one declared option; pass it at least twice\n"
            "  --no-verify       skip the container checksum\n"
            "  --info            print the model geometry and exit\n"
            "  --version         print the library version and numeric backend\n"
            "  -h, --help        print this help\n\n"
            "Output is one line of JSON. Exit status: 0 on success, 1 when the\n"
            "model cannot be opened or the decision fails, 2 on a usage error.\n"
            "See blink(1) or docs/CLI.md.\n",
            program, program, program);
}

static void print_escaped(const char *text)
{
    for (const char *c = text; *c; ++c) {
        switch (*c) {
        case '"': fputs("\\\"", stdout); break;
        case '\\': fputs("\\\\", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
            if ((unsigned char)*c < 0x20u) {
                printf("\\u%04x", (unsigned char)*c);
            } else {
                putchar(*c);
            }
        }
    }
}

static int print_info(const blink_model *model)
{
    blink_model_info info;
    blink_model_get_info(model, &info);
    blink_limits limits = {0u, 0u, 16u, 0u};
    printf("{\"name\":\"%s\",\"width\":%u,\"blocks\":%u,\"ffn_width\":%u,"
           "\"rank\":%u,\"heads\":%u,\"conv_width\":%u,\"stride\":%u,"
           "\"bigram_buckets\":%u,\"max_state\":%u,\"max_question\":%u,"
           "\"max_option\":%u,\"temperature\":%.6f,\"parameters\":%llu,"
           "\"weights_bytes\":%llu,\"session_arena_bytes\":%zu}\n",
           info.name, info.width, info.blocks, info.ffn_width, info.rank,
           info.heads, info.conv_width, info.stride, info.bigram_buckets,
           info.max_state, info.max_question, info.max_option,
           (double)info.temperature, (unsigned long long)info.parameters,
           (unsigned long long)info.weights_bytes,
           blink_session_size(model, &limits));
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "--version") == 0) {
        /* the library's own string, and the kernel it runs on this CPU */
        printf("blink %s (%s)\n", blink_version(), blink_backend());
        return 0;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout, argv[0]);
        return 0;
    }
    const char *path = argv[1];
    const char *state = NULL;
    const char *question = "";
    const char *options[MAX_OPTIONS];
    size_t lengths[MAX_OPTIONS];
    uint32_t count = 0;
    int verify = 1;
    int info_only = 0;

    for (int i = 2; i < argc; ++i) {
        const int has_value = i + 1 < argc;
        if (strcmp(argv[i], "--state") == 0 && has_value) {
            state = argv[++i];
        } else if (strcmp(argv[i], "--question") == 0 && has_value) {
            question = argv[++i];
        } else if (strcmp(argv[i], "--option") == 0 && has_value) {
            if (count == MAX_OPTIONS) {
                fprintf(stderr, "at most %d options\n", MAX_OPTIONS);
                return 2;
            }
            options[count] = argv[++i];
            lengths[count] = strlen(options[count]);
            ++count;
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            verify = 0;
        } else if (strcmp(argv[i], "--info") == 0) {
            info_only = 1;
        } else {
            fprintf(stderr, "unexpected argument: %s\n", argv[i]);
            usage(stderr, argv[0]);
            return 2;
        }
    }

    blink_status status = BLINK_OK;
    blink_model *model = blink_model_open_file(path, verify, &status);
    if (!model) {
        fprintf(stderr, "cannot open %s: %s\n", path,
                blink_status_string(status));
        return 1;
    }
    if (info_only) {
        const int code = print_info(model);
        blink_model_close(model);
        return code;
    }
    if (!state || count < 2u) {
        fprintf(stderr, "a decision needs a --state and at least two --option\n");
        blink_model_close(model);
        return 2;
    }

    blink_limits limits = {0u, 0u, MAX_OPTIONS, 0u};
    blink_session *session = blink_session_create(model, &limits, &status);
    if (!session) {
        fprintf(stderr, "session: %s\n", blink_status_string(status));
        blink_model_close(model);
        return 1;
    }

    float probabilities[MAX_OPTIONS];
    blink_result result;
    status = blink_decide(session, state, strlen(state), question,
                          strlen(question), options, lengths, count,
                          probabilities, &result);
    if (status != BLINK_OK) {
        fprintf(stderr, "decide: %s\n", blink_status_string(status));
        blink_session_free(session);
        blink_model_close(model);
        return 1;
    }

    printf("{\"options\":[");
    for (uint32_t i = 0; i < count; ++i) {
        printf("%s{\"text\":\"", i ? "," : "");
        print_escaped(options[i]);
        printf("\",\"probability\":%.6f}", (double)probabilities[i]);
    }
    printf("],\"argmax\":%u,\"confidence\":%.6f,\"margin\":%.6f,"
           "\"entropy\":%.6f,\"context_positions\":%u,"
           "\"encode_seconds\":%.9f,\"head_seconds\":%.9f}\n",
           result.argmax, (double)result.confidence, (double)result.margin,
           (double)result.entropy, result.context_positions,
           result.encode_seconds, result.head_seconds);

    blink_session_free(session);
    blink_model_close(model);
    return 0;
}
