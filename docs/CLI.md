# The `blink` command

`blink` answers one question of the form *"given this state and this
criterion, which of these options?"* It opens a `.blink` container, encodes
the state, the question and every option in one forward pass, and prints one
probability per option as one line of JSON. It does not generate text: the
options are the only possible answers.

`make` builds it as `build/blink`, and `make install` installs it with its man
page, so `man blink` shows this reference in the terminal. It links libblink
statically and needs nothing else at run time.

```
blink MODEL.blink --state TEXT [--question TEXT] --option TEXT --option TEXT [--option TEXT ...] [--no-verify]
blink MODEL.blink --info [--no-verify]
blink --version
blink -h | --help
```

## Options

| option | meaning |
|---|---|
| `--state TEXT` | The unstructured state to decide about: a ticket, a log line, a JSON document. Required for a decision. It may be empty only when a question is given, because a decision needs at least one byte of context. |
| `--question TEXT` | The criterion, chosen at call time. Optional; empty by default. |
| `--option TEXT` | One declared option. Give it at least twice and at most 32 times; options are scored in the order given. An option must not be empty. |
| `--no-verify` | Skip the CRC-32 check of the container when opening it. |
| `--info` | Print the model's geometry as JSON and exit, without scoring anything. |
| `--version` | Print the library version and the numeric backend it runs on this CPU, e.g. `blink 0.1.0 (neon)`, and exit. |
| `-h`, `--help` | Print a usage summary on standard output and exit. |

The probabilities are conditional on the options given and sum to one. Text
is read as raw bytes, so any encoding works. State, question and each option
have byte limits set by the model, which `--info` reports as `max_state`,
`max_question` and `max_option`. Longer input is refused, not truncated.

The backend `--version` reports is one of `scalar`, `neon`, `sse2`, `avx2` or
`accelerate`. For a W8A8 build (`make W8A8=1`) it is one of `w8a8-scalar`,
`w8a8-sdot`, `w8a8-i8mm`, `w8a8-sse2` or `w8a8-avx2`.

## Output

A decision prints one JSON object:

```json
{"options":[{"text":"delivery and logistics","probability":0.559399},
            {"text":"billing and payments","probability":0.048163},
            {"text":"account access and sign-in","probability":0.392438}],
 "argmax":0,"confidence":0.559399,"margin":0.166960,"entropy":0.838115,
 "context_positions":24,"encode_seconds":0.000111000,"head_seconds":0.000007000}
```

(one line in the real output; wrapped here)

| field | meaning |
|---|---|
| `options` | One object per option, in the order given, with its `text` and its `probability`. |
| `argmax` | Index of the most probable option, counting from zero. |
| `confidence` | Probability of that option. |
| `margin` | Its probability minus the runner-up's. |
| `entropy` | Entropy of the distribution, in nats. |
| `context_positions` | Pooled positions of state and question the options attended over. |
| `encode_seconds`, `head_seconds` | Time spent encoding and in the option head. |

`--info` prints the model's `name`, its geometry (`width`, `blocks`,
`ffn_width`, `rank`, `heads`, `conv_width`, `stride`, `bigram_buckets`), its
byte limits (`max_state`, `max_question`, `max_option`), the calibration
`temperature`, `parameters`, `weights_bytes`, and the `session_arena_bytes` a
session with 16 options needs.

## Environment

| variable | effect |
|---|---|
| `BLINK_X86_SIMD` | On x86-64, `scalar`, `sse2` or `avx2` forces that kernel instead of the one chosen for the CPU. |
| `BLINK_W8A8_KERNEL` | In a W8A8 build, `scalar`, `sdot`, `i8mm`, `sse2` or `avx2` forces that integer kernel. A kernel the CPU or the build lacks is ignored. |

## Exit status

| status | when |
|---|---|
| 0 | Success. |
| 1 | The model could not be opened, a session could not be created, or the decision failed: input over the model's limits, an empty option, or no context at all. The reason is printed on standard error. |
| 2 | A usage error: a missing or unknown argument, fewer than two options or more than 32. |

## Examples

Route a support ticket:

```bash
blink blink-tiny.blink \
  --state "The parcel left the depot on Monday and has not arrived." \
  --question "Which team should handle this ticket?" \
  --option "delivery and logistics" \
  --option "billing and payments" \
  --option "account access and sign-in"
```

Take the answer in a shell script, escalating when the model is unsure:

```bash
blink model.blink --state "$ticket" --option refund --option replace |
  jq -r 'if .confidence < 0.8 then "escalate" else .options[.argmax].text end'
```

For many rows at once, use `eval/evaluate.py`, which drives the same library
and reuses a cached state across rows that share one.

This page, the man page `docs/blink.1` and `usage()` in `tools/blink.c` are
kept in step; `tests/python/test_harness.py` checks that every option appears
in all three.
