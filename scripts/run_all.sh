#!/usr/bin/env bash
# Reproduce everything docs/RESULTS.md reports, from a clean checkout.
#
#   bash scripts/run_all.sh                   the published protocol (~2 h, MPS)
#   bash scripts/run_all.sh --small           also blink-small (~5 h more)
#   bash scripts/run_all.sh --smoke           every stage on a tiny corpus, minutes
#   bash scripts/run_all.sh --skip-download   no network: no WANLI
#
#   BLINK_DEVICE=cpu   training device (default auto: MPS, then CUDA, then CPU)
#   BLINK_SEEDS="7"    fewer runs (default "7 17 27")
#   BLINK_PYTHON=...   interpreter (default .venv/bin/python)
#
# The published protocol is three independent runs of each model -- on MPS a
# seed does not pin a run, see docs/REPRODUCE.md -- named <preset>-<corpus>-s<seed>:
#
#   blink-tiny   synthetic  60 epochs, lr 3e-3
#   blink-tiny   WANLI      25 epochs, lr 3e-3, byte limits from the corpus
#   blink-small  synthetic 100 epochs, lr 2e-3          (--small)
#   blink-small  WANLI      60 epochs, lr 2e-3, limits from the corpus (--small)
#
# Each stage writes machine-readable output under results/, the measured
# sections of docs/RESULTS.md are regenerated from it, and results/MANIFEST.json
# records the checksum of every artefact, so a later run can be compared with a
# published one line by line.
#
# --smoke writes only under data/smoke, artifacts/smoke and results/smoke, and
# touches neither the published results nor the documents. It checks that the
# pipeline runs end to end, not that it reaches any accuracy.
#
# The comparisons against jevlike and on Wikispeedia need a jevlike checkout and
# are not part of this script; scripts/experiments/README.md lists them.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

SMOKE=0 SMALL=0 SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --small) SMALL=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

PYTHON="${BLINK_PYTHON:-$ROOT/.venv/bin/python}"
DEVICE="${BLINK_DEVICE:-auto}"
SEEDS="${BLINK_SEEDS:-7 17 27}"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

if [ "$SMOKE" = 1 ]; then
  DATA=data/smoke ARTIFACTS=artifacts/smoke RESULTS=results/smoke
  SEEDS="${BLINK_SEEDS:-7}"
else
  DATA=data ARTIFACTS=artifacts RESULTS=results
fi
mkdir -p "$DATA" "$ARTIFACTS" "$RESULTS/predictions"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

if [ ! -x "$PYTHON" ]; then
  echo "No interpreter at $PYTHON." >&2
  echo "Create one with:  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy pytest" >&2
  exit 1
fi

# epochs for one training job; --smoke shrinks every one of them to one or two
epochs() { if [ "$SMOKE" = 1 ]; then echo 2; else echo "$1"; fi; }

train() {  # label corpus preset epochs learning-rate [extra...]
  local label="$1" corpus="$2" preset="$3" n="$4" lr="$5"; shift 5
  step "train blink-$label"
  "$PYTHON" -m blink_train.train "$DATA/$corpus/train.jsonl" \
    --validation "$DATA/$corpus/validation.jsonl" \
    --preset "$preset" --epochs "$(epochs "$n")" --learning-rate "$lr" \
    --device "$DEVICE" --seed "${label##*-s}" \
    --output "$ARTIFACTS/blink-$label.blink" \
    --checkpoint "$ARTIFACTS/blink-$label.pt" \
    "$@" >"$RESULTS/train-$label.jsonl" 2>&1
  if grep -q temperature_pinned "$RESULTS/train-$label.jsonl"; then
    echo "  warning: the temperature fit hit its bound; read $RESULTS/train-$label.jsonl" >&2
    if [ "$SMOKE" = 1 ]; then
      echo "  (expected after two epochs on a small corpus)" >&2
    fi
  fi
  return 0
}

evaluate() {  # label model data [extra...]
  local label="$1" model="$2" data="$3"; shift 3
  [ -f "$model" ] && [ -f "$data" ] || { echo "-- skip $label"; return 0; }
  step "evaluate $label"
  "$PYTHON" eval/evaluate.py "$model" "$data" --quiet \
    --json "$RESULTS/eval-$label.json" \
    --predictions "$RESULTS/predictions/$label.jsonl" "$@"
}

# --------------------------------------------------------------- 1. build
step "build the C runtime and its tests"
make clean >/dev/null
make
make fixtures
make test

# ---------------------------------------------------------------- 2. data
step "synthetic corpus"
if [ "$SMOKE" = 1 ]; then
  "$PYTHON" scripts/make_data.py --output "$DATA/synthetic" \
    --train-groups 400 --validation-groups 100 --test-groups 100 \
    >"$RESULTS/data-synthetic.json"
else
  "$PYTHON" scripts/make_data.py --output "$DATA/synthetic" >"$RESULTS/data-synthetic.json"
fi

if [ -d ../SemIf ]; then
  step "external fixtures"
  "$PYTHON" eval/import_fixtures.py --source ../SemIf --output "$DATA/fixtures" \
    >"$RESULTS/data-fixtures.json" || echo "  (skipped)"
fi

WANLI=0
if [ "$SMOKE" = 0 ] && [ "$SKIP_DOWNLOAD" = 0 ]; then
  step "WANLI"
  if "$PYTHON" eval/build_wanli.py --output "$DATA/wanli" >"$RESULTS/data-wanli.json"; then
    WANLI=1
    if [ -d ../SemIf ]; then
      step "SemIf's 256 WANLI rows"
      "$PYTHON" eval/build_wanli256.py --source ../SemIf --output "$DATA/wanli256" \
        >"$RESULTS/data-wanli256.json" || echo "  (skipped)"
    fi
  else
    echo "  (skipped: no network?)"
  fi
fi

# ------------------------------------------------------- 3. train and export
for seed in $SEEDS; do
  train "tiny-synthetic-s$seed" synthetic tiny 60 3e-3
  if [ "$WANLI" = 1 ]; then
    train "tiny-wanli-s$seed" wanli tiny 25 3e-3 --limits-from-data
  fi
  if [ "$SMALL" = 1 ]; then
    train "small-synthetic-s$seed" synthetic small 100 2e-3
    if [ "$WANLI" = 1 ]; then
      train "small-wanli-s$seed" wanli small 60 2e-3 --limits-from-data
    fi
  fi
done

# ------------------------------------------------------------- 4. parity
step "Python tests: C against the float64 reference"
"$PYTHON" -m pytest tests/python -q

# ------------------------------------------------------------ 5. accuracy
presets=tiny
if [ "$SMALL" = 1 ]; then presets="tiny small"; fi
for seed in $SEEDS; do
  for preset in $presets; do
    evaluate "$preset-synthetic-s$seed" "$ARTIFACTS/blink-$preset-synthetic-s$seed.blink" \
      "$DATA/synthetic/test.jsonl"
    evaluate "$preset-wanli-s$seed" "$ARTIFACTS/blink-$preset-wanli-s$seed.blink" \
      "$DATA/wanli/test.jsonl"
    evaluate "$preset-wanli256-s$seed" "$ARTIFACTS/blink-$preset-wanli-s$seed.blink" \
      "$DATA/wanli256/test.jsonl" --bootstrap 2000
  done
  # Transfer: a model trained on the synthetic corpus, asked for something else.
  for fixture in authored144 perturbations108; do
    evaluate "tiny-synthetic-s$seed-$fixture" "$ARTIFACTS/blink-tiny-synthetic-s$seed.blink" \
      "$DATA/fixtures/$fixture.jsonl" --bootstrap 500
  done
done
# Untrained weights must land on chance; if they do not, the harness is
# measuring itself.
for preset in tiny small; do
  evaluate "$preset-untrained" "build/synth-$preset.blink" "$DATA/synthetic/test.jsonl" \
    --skip-controls --bootstrap 0
done

if [ "$SMOKE" = 1 ]; then
  step "benchmarks (not saved)"
  make bench >"$RESULTS/bench.jsonl"
  head -3 "$RESULTS/bench.jsonl"
  printf '\n\033[1msmoke run complete.\033[0m Everything it wrote is under %s, %s and %s.\n' \
    "$DATA" "$ARTIFACTS" "$RESULTS"
  exit 0
fi

# ----------------------------------------------------------- 6. benchmarks
# Latency, memory and the scalar build for the SIMD comparison. Run it on an
# idle machine: every other process shows up in the percentiles.
step "benchmarks"
make bench-save

# ------------------------------------------------------------ 7. document
step "regenerate the measured sections of docs/RESULTS.md"
"$PYTHON" scripts/report_results.py --write

step "manifest and summary"
"$PYTHON" scripts/manifest.py >results/MANIFEST.json
"$PYTHON" scripts/summarise.py >results/SUMMARY.md

printf '\n\033[1mdone.\033[0m results/ holds every report; results/MANIFEST.json holds every checksum.\n'
