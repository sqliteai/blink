# Third-party sources

Blink's own code is Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Nothing in this repository vendors third-party code or weights. What follows is everything the project
reads, cites or was informed by, with its own terms.

## Related systems and external baselines

**Jev and "System One Models" — TypeSafe (closed).**
<https://typesafe.ai/blog/introducing-system-one-models-and-jev>
The interface Blink offers is described in TypeSafe's public material. No
Jev weights, code or endpoint were used. Every Jev figure quoted in this
repository is taken from published material and is labelled as such. Jev,
TypeSafe and related names are the property of their owners. This project is
independent and unaffiliated.

**jevlike — MIT.**
<https://github.com/vinnylarouge/jevlike>
An independent one-pass choice model in PyTorch, measured here as an external
baseline. `eval/compare_external.py` converts Blink rows into its row format
and scores its checkpoints through Blink's metric code, so the two systems are
measured by identical arithmetic; `eval/bench_external.py` measures its latency
and memory on the same machine as Blink's benchmarks. It is installed as a
dependency for those comparisons and is not vendored. No code was copied.

**SemIf — MIT. Copyright (c) 2026 TheoLeeCJ.**
<https://github.com/TheoLeeCJ/SemIf>
An independent generation-free readout from frozen open models, used here as
an external baseline. Its figures are quoted from its own published results.

`eval/build_wanli256.py` rebuilds the 256-row WANLI subset SemIf evaluates,
from the row identifiers committed in
`benchmarks/manifests/source-selection.jsonl`. Only the identifiers are read;
the rows themselves come from WANLI at its pinned revision.

`eval/import_fixtures.py` converts two of SemIf's owned fixtures,
`benchmarks/data/authored144.jsonl` and `benchmarks/data/perturbations108.jsonl`,
into Blink rows. The conversion copies state, question, option descriptions and
labels verbatim and changes nothing else; both the source and the converted
checksums are recorded in `data/fixtures/manifest.json`. The fixtures are used
under SemIf's MIT licence and are **not** committed here — the importer needs a
SemIf checkout.

## Teacher model (branch blink-distilled)

**MiniCPM5-2B — OpenBMB, Apache-2.0.**
<https://huggingface.co/openbmb/MiniCPM5-2B>
Pinned to revision `12a3808a956f869c767195e9266b59c4d21d92e2`.

Used only as a teacher for distillation: `scripts/teacher_label.py` runs it
locally with MLX and records its probability over each row's options as a
training target. Its weights are downloaded into `artifacts/external/`, which
is not committed, and nothing from the model is redistributed. The containers
trained with its targets contain Blink's own weights, trained on those
probabilities; they are marked as distilled wherever they are reported. The
labelled corpus is not committed either (it derives from WANLI, see below);
`results/teacher-wanli-minicpm5.json` records its checksums.

- **MLX and mlx-lm** — MIT. Only in the separate `.venv-teacher` environment
  that runs the teacher; not a training or runtime dependency.

## Datasets

**WANLI — CC-BY-4.0.**
<https://huggingface.co/datasets/alisawuffles/WANLI>
Pinned to revision `61c95318fd71c55b6ba355d76253254615f387ec`.

> Alisa Liu, Swabha Swayamdipta, Noah A. Smith, Yejin Choi.
> *WANLI: Worker and AI Collaboration for Natural Language Inference Dataset
> Creation.* Findings of EMNLP 2022.

`eval/build_wanli.py` downloads it at that revision into `data/.cache` and
derives decision rows from it: premise to state, hypothesis to question, and
the three NLI relations to three described options. Neither the original data
nor the derived rows are committed. Review the dataset card for the terms that
apply to your use.

## Tools

- **PyTorch** — BSD-3-Clause. Training only; it produces no reported number.
- **NumPy** — BSD-3-Clause. The float64 reference and the data builders.
- **pytest** — MIT. The parity suite.

None of these are needed to build, test, benchmark or run the C runtime.

- **Accelerate** — Apple's system framework, part of macOS. Linked only by the
  optional `make ACCELERATE=1` build; nothing from it is redistributed.

## What this repository contains

- `src/`, `include/`, `tools/`, `bench/`, `tests/`, `examples/` — original C.
- `python/blink_train/`, `eval/`, `scripts/` — original Python.
- `data/synthetic/` — generated locally by a seeded generator in this
  repository. No external text.
- `artifacts/*.blink` — trained here from the above.

No model weights from any other project are redistributed.
