# Architecture

This document has two halves. The first is what TypeSafe published about Jev
and where Blink sits among open systems with the same interface. The second is
what Blink does and why each choice was made.

---

## 1. Context

### 1.1 Jev and "System One Models" (TypeSafe, closed)

TypeSafe describes Jev as the first of a class they call *System One Models*:
models built to make fast structured decisions that software consumes
directly, rather than to converse. From their public material:

| Claim | What it means mechanically |
|---|---|
| "all outputs in a single query" | no autoregressive loop; the answer is read out of one forward pass |
| "type-safe structured values" | the output space is declared in advance, so a schema violation is impossible by construction rather than by validation |
| calibrated probabilities with every output | the model reports confidence, and higher confidence is supposed to mean higher accuracy |
| RLCD, "Reinforcement Learning for Calibrated Decisions" | the training objective targets calibration rather than human preference, in contrast with RLHF |
| 70 ms – 500 ms end to end, input at \$0.042/MTok, output free | "output free" is the giveaway: there are no output tokens to charge for |

Nothing about the parameter count, the layer structure or the training data is
published. The architecture below is not a reconstruction of Jev. What is
reproducible from the public description is the **interface**: unstructured
state, plus a criterion defined at call time, plus a declared option set, in,
and one probability per option out, in a single pass.

### 1.2 Where Blink sits

Two independent open projects expose the same interface and are used in this
repository as external baselines: [jevlike](https://github.com/vinnylarouge/jevlike),
a small PyTorch model trained from scratch, and
[SemIf](https://github.com/TheoLeeCJ/SemIf), which reads option logits out of
a frozen open LLM (Qwen3.5-4B and smaller). The measured comparison, speed and
memory included, is in
[RESULTS.md](RESULTS.md#comparison-with-other-systems).

|  | jevlike | SemIf | Blink |
|---|---|---|---|
| weights | 45k / 398k params, fp32 | 0.6B–4B, frozen | 407k / 7.9M params, int8 |
| footprint | Python + torch | 0.6–3.0 GB | ~450 KB / ~7.9 MB, mapped read-only |
| runtime | Python | Python / llama.cpp / MLX | C99, no dependencies beyond libc |
| allocation while scoring | GC | GC | none |
| state reuse | none | approximate KV prefix cache | exact, by construction |
| semantic strength | weak | strong | in between, and measured |

Neither is embeddable: one is a Python research model, the other needs a GPU
that can hold a 4B model. Blink is the same interface with a runtime you can
link into a C program, a daemon or a device.

---

## 2. What Blink does

### 2.1 The interface

```c
blink_state_set(session, state, length);          /* encode once   */
blink_menu_set(session, options, lengths, count); /* encode once   */
blink_score(session, question, length, probs, &result);  /* cheap  */
```

Three inputs, one output. The option list is read at call time, so the model
has no notion of a fixed label set: `count` and the option text may change on
every call. The output is a probability per declared option, so a schema
violation is not possible — there is no string to parse and nothing to repair.

The split into three calls is the architecture, not a convenience. It is what
makes state reuse possible, and the reuse is **exact**: because the state
and the question are encoded as two independent sequences whose hidden states
are only ever combined inside the attention, reusing an encoded state produces
bit-for-bit the same floats as re-encoding it. `tests/c/test_session.c` checks
that with `memcmp`, not with a tolerance.

### 2.2 The forward pass

```
                     state bytes        question bytes       option bytes
                          │                   │                   │
  unigram + hashed bigram + position + segment embedding  ────────┤
                          │                   │                   │
             ┌────────────┴───────────────────┴───────────────────┴──────┐
             │  stem: depthwise conv (k=5) + pointwise projection        │  full byte
             └────────────┬───────────────────┬───────────────────┬──────┘  resolution
                          │                   │                   │
                   mean-pool by `stride` (4 for tiny, 8 for small)
                          │                   │                   │
             ┌────────────┴───────────────────┴───────────────────┴──────┐
             │  N blocks: LN → conv + global mean → LN → FFN             │  reduced
             │            (the last few also carry self-attention)       │  resolution
             └────────────┬───────────────────┬───────────────────┬──────┘
                          │                   │                   │
                    K, V ◄┴──────────┐        │         mean-pool ► option
                          │          │ question summary            vectors
                          │          └────────► FiLM ◄─────────────┘
                          │                      │
                          │                      ▼  Q
                          └────────► multi-head cosine option attention
                                              │
                                  logit / temperature → softmax
```

**Byte level, no tokenizer.** The input is raw bytes. Program state is not
curated text: it contains JSON, identifiers, UTF-8 from any language, and
sometimes NUL bytes. A tokenizer would be a second artefact to ship, a second
thing to keep in sync, and a source of boundary bugs (appending an answer
token can change how the prompt before it is tokenized). Bytes have none of
those problems. `tests/c/test_inputs.c` checks that all 256 byte values, embedded
NULs and invalid UTF-8 all go through.

**Hashed byte bigrams.** A pure per-byte embedding is close to useless: `e`
carries no information about whether it is in `refund` or `screen`. Each
position therefore also looks up a learned embedding for its `(previous,
current)` byte pair, hashed into a fixed table. This is one extra gather and
add per byte — no matmul — and it is where most of the tiny preset's parameters
live. The hash is a fixed 32-bit mix defined once in `blink_bigram_index` and
checked against the NumPy and pure-Python implementations for all 65,536 pairs
in `tests/python/test_parity.py`.

**A stem, then a pool, then the blocks.** This is the main efficiency
decision. The expensive part of a byte-level encoder is running a
feed-forward at every byte. Blink runs one cheap full-resolution stem — a
depthwise convolution for local n-grams and a pointwise projection for channel
mixing — then mean-pools non-overlapping windows of `stride` positions, and
only then runs the feed-forward blocks. A 512-byte state becomes 64 positions
before any block executes. The measured effect is in [RESULTS.md](RESULTS.md);
the structural effect is that the key/value cache, the attention buffer and the
block compute all shrink by the stride.

**Blocks are O(L·W), never O(L²·W).** Each block mixes locally with a
depthwise convolution and globally with a single masked mean projected back to
every position. There is no self-attention anywhere in the encoder. The only
attention in the model is the option head, which is O(N·L·R) and tiny because
N is a handful and L is already pooled. This is what makes the cost linear in
the state length instead of quadratic.

**ReLU² rather than GELU or SiLU.** The activation must be identical in
PyTorch, in the NumPy reference and in C. `max(x,0)²` is a compare and a
multiply — exactly representable everywhere, with no transcendental
approximation to disagree about. A GELU would have made the parity test a
question of which `erf` approximation each implementation used.

**Segment embeddings.** One encoder handles all three roles. A learned
per-segment vector (state, question, option) is added at embedding time so the
head can tell a state position from a question position even though they went
through the same weights.

**One self-attention layer, and only after the pool.** Everything described so
far is linear in the sequence length, which is fast but means two specific
distant positions can never be related to each other: a depthwise convolution
reaches a few neighbours, and a masked mean reaches everything without
distinguishing anything. The trailing `mixer_blocks` blocks therefore carry a
self-attention sub-layer over the **pooled** positions. At stride 8 a 512-byte
state is 64 positions, so the quadratic term is small next to the block
feed-forward; at full byte resolution it would have been the dominant cost.
This is the second reason the pool exists.

**The question reads the state; the state never reads the question.** Encoding
the two as separate sequences is what makes a cached state exactly reusable,
but it has a cost that is easy to miss: nothing inside the encoder ever
compares a question byte against a state byte. They meet only in the option
head, which takes a weighted *average* over both — and an average cannot
express *"are these the same word"*.

That is not an abstract concern. Deciding whether *"the order was cancelled"*
supports or contradicts *"the order for Leila\'s keypad was cancelled on
Tuesday morning"* is exactly that comparison, over a pair of sentences that
differ in one verb. Without a path for it, half of all training runs failed to
find one and collapsed the two answers together; see
[RESULTS.md](RESULTS.md#judgment-and-question-to-state-attention).

So the question gets one attention layer over the state, after its own blocks.
The direction is the whole design: because the state attends to nothing, its
encoding — and the keys and values this layer derives from it — remain a
function of the state alone, computed once in `blink_state_set` and reused by
every later question. The exact-cache property survives, and
`tests/c/test_session.c` still checks it with `memcmp`.

It costs about 25% more on the cached path at the tiny preset, and it took the
across-seed standard deviation of `judgment` from 0.182 to 0.0015.

**The option head.** Each option queries the encoded context and is scored
against what it reads back. Several heads, with every vector made unit length
(û = u / ‖u‖):

```
for each head h:
    a_n = softmax_l( √d · q̂_n,h · k̂_l,h )
    c_n = normalise( Σ_l a_n,l · v_l,h )
    logit_n += q̂_n,h · c_n                   # a cosine, in [−1, 1]
logit_n *= logit_scale
p = softmax_n(logit_n / temperature)
```

**Why cosines.** With plain dot products nothing bounds a logit: a run can
grow ‖q‖, ‖k‖ and ‖v‖ to sharpen attention and raise its confidence, and the
loss rewards it on training data. One blink-small run did exactly that — the
head's projections grew twelve-fold and its validation NLL reached 160 — and
the temperature fit then sat at its upper bound trying to undo it. With unit
vectors every logit is bounded by `heads × |logit_scale|` by construction, so
confidence can only come from the one scalar the temperature fit already
controls. `tests/python/test_parity.py` inflates the QKV projections a
thousand-fold and checks that the logits stay inside that bound. Keys are
normalised once, when the state is encoded, so the cached path pays nothing
for it. The change moved blink-tiny from 0.6064 ± 0.0149 to 0.6334 ± 0.0171
and ended the blink-small divergence; it is container format 3.

Separate heads with separate projections are several reads of different
subspaces, not one read repeated. Nothing in the head is indexed by option
identity, so the menu stays runtime-defined.

**The query is conditioned on the question, and that is the change that
matters.** If `q_n` depends on the option text alone, the question still
reaches the score, but only by one indirect route: it contributes keys
and values to the shared context, so it can change *where* an option looks and
what that position returns.

```
logit_n = q_n · Σ_l softmax_l( q_n · k_l ) v_l     with l over state + question
```

That route is real but weak, and the reason is worth being precise about. For
the question to select between options, the option's own query has to
preferentially attend to the question positions that matter — but `q_n` was
built from the option text and knows nothing about the question, so it has no
reason to. The model can learn to route *every* option's attention the same
way, which the softmax over options then discards; what it has no direct
mechanism for is making one option's reading of the state depend on the
question.

This is a claim about how hard something is to learn, not a proof that it
cannot be represented. The evidence that it matters in practice is the ablation
in [RESULTS.md](RESULTS.md), not this paragraph.

Two of the four synthetic task families were chosen to probe exactly this:

- *Which city does Omar live in?* Every candidate city appears in the state. The
  option `Rennes` can attend to where `Rennes` occurs, but whether that is the
  right answer depends on `Omar`, which lives only in the question.
- *Which product scored highest — or lowest?* The values are in the state, but
  the direction of the comparison is in the question, and flipping it must flip
  the ranking of every option.

Blink therefore modulates the option vectors with a summary of the question
before projecting them, in the manner of FiLM:

```
s      = mean of the question's pooled hidden states
γ, β   = W_γ · LN(s),  W_β · LN(s)
q_n    = W_q · LN( o_n · (1 + γ) + β )
```

An empty question gives `s = 0`, hence `γ = β = 0`, and the option vectors pass
through untouched — the degenerate case is the old behaviour, not a special
case in the code.

Now the question reaches the score twice: through the context, as before, and
through a direct multiplicative term on every option's query.

This costs two extra matrix products per decision — not per option — plus one
elementwise pass per option, which is negligible next to encoding the question.
It does move the option-vector-to-query projection out of `blink_menu_set` and
into `blink_score`, so what the menu cache holds is the *encoded option
vectors*, which is the expensive part; the cheap projection is redone whenever
the question changes.

**The measurement does not support this argument, and the argument does not
get to stand in for one.** A three-seed ablation
([RESULTS.md](RESULTS.md#the-film-ablation-still-undecided)) is *undecided* on every
slice: the spread between seeds within either arm is larger than the gap
between the arms. Two earlier single-seed runs appeared to show FiLM rescuing
first `judgment` and then `comparison`; both were noise, and both were reported
here before the seeds were run.

So: the conditioning is cheap, it is motivated, it makes the degenerate case
(empty question) exactly the old behaviour, and at this scale it has no
demonstrated effect. It stays in the architecture because the argument about
the functional form is sound and the cost is two matrix products per decision,
not because the numbers earned it. If you need the question to select between
options, this is the mechanism that makes it expressible; whether a
390k-parameter model can then learn it is a separate question that these
experiments do not answer.

Two practical notes, both learned the hard way:

- `film_gamma` and `film_beta` are **zero-initialised**, so the modulation
  starts as an exact identity. Initialised randomly they perturb a working
  residual path from the first step; one 100-epoch run spent twenty epochs in a
  worse optimum than it started in.
- They stay **fp32**, outside the quantized set. They scale the option vectors,
  so their quantization error is multiplicative and compounds through the head.
  Quantizing them drove validation NLL from 0.87 to 1.51 when
  quantization-aware training switched on. The exemption costs 24 KiB at the
  tiny preset.

**Order equivariance is a property, not a hope.** Because each option's logit
is computed independently and the only interaction is the final softmax,
permuting the menu permutes the probabilities exactly. `tests/c/test_inputs.c`
checks it, and the same test checks that the ratio of two options'
probabilities does not change when other options are added or removed — which
is what makes the softmax a conditional score over the supplied alternatives
rather than an arbitrary ranking.

### 2.3 Calibration

Blink does not implement RLCD; TypeSafe has not published it. It does the
honest, checkable version of the same intent:

- The training objective is cross-entropy over the option set, which is a
  strictly proper scoring rule. Minimising it rewards honest probabilities, not
  only the right argmax.
- A single temperature is fitted after training by minimising validation NLL,
  and baked into the container. The fit is bounded to [0.05, 20]; if it lands
  on a bound, training prints a `temperature_pinned` warning and records it in
  its log, because a pinned temperature means the logits were not something a
  single scale could correct — the symptom of the divergence described under
  the option head.
- Calibration is then **reported on the test split**, which the temperature was
  never fitted on: ECE, MCE, a reliability table, Brier, NLL, and a
  risk-coverage curve showing what accuracy is bought by refusing the
  low-confidence cases.

The probability Blink returns is conditional on the options it was given. That
caveat is repeated in
[METHOD.md](METHOD.md#interpretation-rules) rather than buried.

### 2.4 Quantization and the container

Weights ship as symmetric per-row int8 with one fp32 scale per output row.
Layer-norm gains and biases, the depthwise convolutions and the segment table
stay fp32: together they are a few hundred values and quantizing them costs
accuracy for no measurable space. The last 30% of training runs with
straight-through fake quantization, so the exported model behaves like the
trained one; the export reports the measured round-trip error rather than
assuming it is small.

Activations are fp32 and accumulate in fp32 in the default build, which is
what makes it exactly reproducible against a float64 reference. That W8A32
product is the normative definition.

`make W8A8=1` builds the optional W8A8 variant. Each input row of a projection
gets its own symmetric int8 scale (`m = max|x|`, `q = round_half_even(x ·
127/m)`, `s_x = m/127`), and the product is an exact int32 sum:

```
out[r] = ((float)(Σ_c w[r][c] · q_c) · scale[r]) · s_x
```

Because integer addition is exact, the three kernels that compute it are
bit-identical to one another: the scalar loop (the definition), SDOT (16
products per instruction, any ARMv8.2 core with the dot-product extension)
and I8MM (SMMLA, a 2×8 by 2×8 block per instruction, chosen at run time when
the CPU reports it, except on Apple silicon, where it measured level with
SDOT and SDOT stays the default). A batch of questions is still bit-identical to the same
questions scored one at a time, a cached state is still exact, and nothing
allocates while scoring. What changes is the model: activations are rounded
to 8 bits, so results differ from the W8A32 definition by more than fp32
rounding. `python/blink_train/reference.py` has the same rounding
(`activations="int8"`), and RESULTS.md measures the effect on every published
model. `blink_backend()` reports which build and kernel is running.

The `.blink` container is a flat header, a tensor table and a 64-byte-aligned
blob, described byte for byte in
[`python/blink_train/container.py`](../python/blink_train/container.py). It is
mapped read-only and **used in place**: no weight byte is ever copied into the
heap, so N processes sharing a model pay for it once. Every geometry field is
validated on open and the whole file is CRC-32 checked;
`tests/c/test_container.c` corrupts 20 individual fields and every
truncation point and requires a specific error for each.

The header carries a format version, currently **3**. Adding a field or a
tensor is backward compatible and does not bump it; changing what an existing
tensor means — its dtype, its shape or the computation it feeds — does,
because an old container would otherwise load and silently compute something
else. Version 2 made FiLM fp32 and added the cross layer; version 3 is the
cosine head. The runtime refuses any other version with `BLINK_E_VERSION`.

### 2.5 Memory

The caller owns everything.

- **Weights**: mapped, shared, never copied.
- **Session**: one contiguous arena whose size is a pure function of the model
  and the declared limits. `blink_session_size` tells you the number;
  `blink_session_init` places the session in a buffer you supply — a static
  array, a stack frame, a pool. Declaring smaller limits buys a smaller arena.
- **Scoring**: no allocation at all. `tests/c/check_no_malloc.sh` asserts
  mechanically that `blink_runtime.o` and `blink_kernels.o` reference no
  allocator symbol, which is why the allocating convenience wrapper lives in
  its own translation unit.

### 2.6 Determinism

The build pins `-ffp-contract=off`, so a fused multiply-add cannot silently
change a result between compilers or architectures. Every kernel is a plain
loop, and `python/blink_train/reference.py` mirrors each loop in float64. The
parity test requires the two to agree to `1e-5` on probabilities; the measured
disagreement across all three presets is under `5e-7`, which is fp32 rounding
and nothing else.

### 2.7 What was deliberately left out

- **Self-attention at full byte resolution.** O(L²) over bytes defeats the
  purpose. Self-attention exists only after the pool, and only in the trailing
  blocks.
- **A tokenizer.** See above.
- **Threads.** A session is single-threaded; concurrency is one session per
  thread over one shared model. That keeps the arena contract simple and the
  measurements honest.
- **W8A8 by default.** It exists (`make W8A8=1`, section 2.4) and is about
  twice as fast, but the default build stays W8A32 because that is the one
  exactly reproducible against the float64 reference.

---

## 3. Honest limitations

- A 390k-parameter byte model is not a language model. It learns the tasks it
  is trained on; it does not arrive knowing what a refund is. The per-family
  breakdown in [RESULTS.md](RESULTS.md) shows exactly which of the four
  synthetic task families it does and does not learn, and the transfer result
  on the external fixtures shows what happens when you ask it for something it was
  never trained on.
- The option list must be complete before scoring. That is inherent to a
  one-pass conditional softmax.
- The reported probabilities are conditional on the supplied options and are
  calibrated on the workload they were fitted on. Recalibrate on yours.
- This is not Jev and does not reproduce Jev's training. No claim here is a
  comparison with a Jev endpoint; the only Jev numbers quoted anywhere in this
  repository are published ones.
