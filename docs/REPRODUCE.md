# Reproduce

Every number in [RESULTS.md](RESULTS.md) comes from `bash scripts/run_all.sh`.
This page is the shorter path to each piece of it.

---

## The C runtime alone

The library, its tests and its benchmarks need no Python and no trained model.
`blink_synth` writes valid containers full of pseudo-random weights, which is
enough to exercise every code path and to measure latency and memory.

```bash
make                     # static library, shared library, tools, benchmarks
make test                # unit tests plus the no-allocation check
make bench               # latency and memory, as JSON on stdout
make ACCELERATE=1 test   # macOS: the Accelerate backend, built in build-accelerate/
./build-accelerate/bench_latency > results/bench-latency-accelerate.jsonl
make W8A8=1 test         # int8 activations with SDOT/I8MM, built in build-w8a8/
bash scripts/x86-docker-test.sh   # x86-64 SIMD paths in a linux/amd64 container
./build-w8a8/bench_latency > results/bench-latency-w8a8.jsonl
```

Expected: every test prints `ok`, and `check_no_malloc` confirms that the
scoring path references no allocator symbol.

The build pins `-std=c99 -O3 -ffp-contract=off`. The last flag is not
decoration: it stops a fused multiply-add from changing results between
compilers, which is what makes the parity test below meaningful. Overriding it
invalidates the parity tolerance.

```bash
make CC=gcc-14                       # a different compiler
make OPT='-O2 -g -fsanitize=address' # sanitised build
```

## The Python environment

Only training, export and the evaluation harness need it.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy pytest
```

Every command below assumes `PYTHONPATH=python`; `scripts/run_all.sh` sets it.

## Data

```bash
PYTHONPATH=python .venv/bin/python scripts/make_data.py            # synthetic
PYTHONPATH=python .venv/bin/python eval/build_wanli.py             # WANLI (downloads)
PYTHONPATH=python .venv/bin/python eval/import_fixtures.py --source ../SemIf
PYTHONPATH=python .venv/bin/python eval/build_wanli256.py --source ../SemIf  # SemIf's 256 rows
```

`scripts/make_data.py` is seeded and writes `data/synthetic/manifest.json` with
the SHA-256 of each split; two runs on any machine produce identical files.
WANLI is pinned to one dataset revision and cached under `data/.cache`.

## Train and export

```bash
PYTHONPATH=python .venv/bin/python -m blink_train.train \
  data/synthetic/train.jsonl \
  --validation data/synthetic/validation.jsonl \
  --preset tiny \
  --output artifacts/blink-tiny-synthetic-s7.blink \
  --epochs 60 --learning-rate 3e-3 --device mps --seed 7
```

That is the published blink-tiny. On WANLI it is 25 epochs with
`--limits-from-data`, which sizes the byte limits from the training corpus and
reports how many rows they cover; `scripts/run_all.sh` has the full protocol,
blink-small included.

`--device` takes `cpu`, `mps` or `cuda`; `auto` picks the best available. On
this machine MPS is roughly fifty times faster than the CPU path for these
shapes, and the two agree to five decimal places on validation NLL.

The trainer prints one JSON object per epoch, fits the calibration temperature
on validation, and writes the `.blink` container plus an export report that
includes the measured int8 round-trip error.

Two flags matter for reproducing the reported accuracy:

- `--epochs`. The `selection` and `comparison` families are not learned
  gradually; they sit at chance for tens of epochs and then improve sharply.
  A short run will report them at chance and will be, in its own terms,
  correct. See the note in [RESULTS.md](RESULTS.md).
- `--qat-fraction`. The final fraction of training runs with straight-through
  int8 rounding, and only those epochs are eligible for checkpoint selection,
  so what is selected is comparable to what is exported.

## Ablations

Each sub-layer can be switched off with one flag, so an ablation is two runs
that differ in exactly that flag:

```bash
PYTHONPATH=python .venv/bin/python -m blink_train.train \
  data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
  --preset tiny --no-film \
  --output artifacts/blink-tiny-nofilm-s7.blink \
  --epochs 60 --learning-rate 3e-3 --device mps --seed 7
```

`--no-film` drops the question conditioning of the option vectors and
`--no-mixer` the self-attention sub-layers. Run each arm several times: on MPS
one run per arm has given three different answers to the same question
(RESULTS.md, *The FiLM ablation*). The queues that produced the published
ablations are in `scripts/experiments/`.

`scripts/make_family_split.py --family selection --output data/selection`
writes a corpus with one task family, which separates a family that is slow to
learn from one that is not learned at all.

## Evaluate

```bash
PYTHONPATH=python .venv/bin/python eval/evaluate.py \
  artifacts/blink-tiny-synthetic-s7.blink data/synthetic/test.jsonl \
  --json results/eval-tiny-synthetic-s7.json \
  --predictions results/predictions/tiny-synthetic-s7.jsonl
```

This runs the **C runtime** through the ctypes binding, not the PyTorch model.
It prints the headline metrics, the controls and the per-family breakdown, and
writes the full report — reliability table, risk–coverage curve, bootstrap
intervals, perturbations — plus one line per row so any aggregate can be
recomputed without rerunning the model.

To regenerate the measured sections of RESULTS.md (§1–§3) from the reports in
`results/`, evaluating any published container whose report is missing:

```bash
PYTHONPATH=python .venv/bin/python scripts/report_results.py --write
```

## External baselines

The speed and memory comparison in the README measures jevlike on the same
machine as `make bench`, one CPU thread, batch one. It needs jevlike installed
in the environment and the checkpoints `scripts/experiments/queue_external.sh`
trains:

```bash
.venv/bin/python eval/bench_external.py artifacts/external/external-w288-s7.pt \
  --json results/bench-external-w288.json
.venv/bin/python eval/bench_external.py artifacts/external/external-w64-s7.pt \
  --json results/bench-external-w64.json
/usr/bin/time -l ./build/blink artifacts/blink-tiny-synthetic-s7.blink \
  --state "..." --question "..." --option a --option b   # Blink's peak RSS
```

SemIf's figures are quoted from its published results and were not rerun here.

To evaluate the Accelerate or W8A8 build with the same harness, point the
ctypes binding at it (`build-w8a8/libblink.dylib` for W8A8; the parity suite
then compares with the reference's int8-activation mode on its own):

```bash
BLINK_LIBRARY=build-accelerate/libblink.dylib PYTHONPATH=python \
  .venv/bin/python eval/evaluate.py artifacts/blink-tiny-synthetic-s7.blink \
  data/synthetic/test.jsonl
```

## x86-64 without an x86 machine

```bash
bash scripts/x86-docker-test.sh                 # tests, parity, timings
bash scripts/x86-docker-test.sh --skip-bench    # correctness only
```

Runs in a linux/amd64 container (`gcc:14` by default, pulled on first use,
about 1.2 GB) with the repository mounted read-only. It runs the C suite for
the default build, W8A8, SIMD compiled out, `-O0` and ASan + UBSan, each with
every x86 level the CPU reports. It checks a trained model's probabilities
at every level against the float64 reference, computed on the host with
`.venv`, and times each level. On Apple silicon the container runs under
Rosetta, which the script reports; its timings are then not x86 timings.

## Parity

```bash
PYTHONPATH=python .venv/bin/python -m pytest tests/python -q
```

Checks that the C runtime agrees with the float64 reference in
`python/blink_train/reference.py` on every preset and a set of awkward inputs
(empty question, empty state, all 256 byte values, invalid UTF-8, inputs
exactly at the limits), that PyTorch agrees with the reference once the export
is read back, that the bigram hash agrees across C, NumPy and pure Python for
all 65,536 byte pairs, and that a container written by the C tool loads in
Python and scores identically.

The tolerance is `1e-5` on probabilities. Observed disagreement is below
`1e-6`, which is fp32 accumulation against float64 accumulation and nothing
else.

## Everything

```bash
bash scripts/run_all.sh
```

About two hours on an Apple M-series laptop with MPS: three runs of blink-tiny
on each corpus, every evaluation, the benchmarks, and the regenerated
document. Switches:

```bash
bash scripts/run_all.sh --smoke             # every stage on a small corpus, minutes
bash scripts/run_all.sh --small             # also blink-small, about five hours more
bash scripts/run_all.sh --skip-download     # no network: no WANLI
BLINK_SEEDS=7 bash scripts/run_all.sh       # one run of each model
BLINK_DEVICE=cpu bash scripts/run_all.sh    # bit-reproducible, much slower
```

`--smoke` writes only under `data/smoke`, `artifacts/smoke` and
`results/smoke`, and leaves the published results and documents alone. It
proves the pipeline runs, not that it reaches any accuracy.

It writes `results/MANIFEST.json` with the SHA-256 of every input, artefact and
report alongside the compiler, its exact command line, the platform and the git
revision, and `results/SUMMARY.md` with the tables. Two runs can be diffed
rather than eyeballed.

## Determinism

`--seed` pins a run on CPU and does **not** pin one on MPS. Measured directly,
two runs of the same seed, corpus and architecture:

| backend | run 1, epoch-1 train NLL | run 2 |
|---|---|---|
| `--device cpu` | `1.3074637349446614` | `1.3074637349446614` |
| `--device mps` | `1.307463773091634` | `1.307463755607605` |

MPS accumulates in an order it does not promise to repeat, and by the end of a
60-epoch run that divergence is worth tenths of a point of accuracy on the
task families that are learned through a sharp transition.

Two consequences, both of which apply to the numbers in
[RESULTS.md](RESULTS.md):

- A "three-seed" result taken on MPS is three independent runs, not three
  seeds. Its standard deviation mixes seed variation with backend
  non-determinism and cannot separate them.
- Reproducing a published figure exactly requires `--device cpu`, which on this
  machine is about fifty times slower. Use it to verify a specific claim, not
  to train.

The C runtime is a different matter: given a container it is deterministic to
fp32 with the pinned flags, and `tests/python/test_parity.py` holds it to
`1e-5` against a float64 reference.

## What is and is not bit-reproducible

| | reproducible |
|---|---|
| data generation | yes, from the seed, on any machine |
| the `.blink` container from a given checkpoint | yes |
| accuracy and calibration from a given container | yes |
| C runtime output from a given container | yes, to fp32, with the pinned flags |
| PyTorch training on CPU | yes, from the seed |
| PyTorch training on MPS or CUDA | no — see [Determinism](#determinism) above |
| wall-clock timings | no — they depend on the machine and its load |

If a training run does not land on the published accuracy, compare
`results/train-*.jsonl` epoch by epoch before suspecting the runtime: the
parity test is the thing that says whether the runtime is faithful, and it is
deterministic.
