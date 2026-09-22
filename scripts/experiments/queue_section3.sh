#!/usr/bin/env bash
# Retrain everything docs/RESULTS.md section 3 reports, on the current
# architecture, three seeds each. Every run also writes its fp32 checkpoint, so
# a future container-format change costs a re-export rather than a retrain --
# a lesson from bumping the format and losing three trained models to it.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python"
PY=.venv/bin/python

run() {
  local label="$1"; shift
  echo "[s3] $label starting $(date +%H:%M:%S)"
  "$PY" -m blink_train.train "$@" > "results/train-$label.jsonl" 2>&1
  echo "[s3] $label exit=$? $(date +%H:%M:%S)"
}

for seed in 7 17 27; do
  run "tiny-synthetic-s$seed" \
    data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
    --preset tiny --output "artifacts/blink-tiny-synthetic-s$seed.blink" \
    --epochs 60 --device mps --learning-rate 3e-3 --seed "$seed"
done

for seed in 7 17 27; do
  run "tiny-wanli-s$seed" \
    data/wanli/train.jsonl --validation data/wanli/validation.jsonl \
    --preset tiny --limits-from-data \
    --output "artifacts/blink-tiny-wanli-s$seed.blink" \
    --epochs 25 --device mps --learning-rate 3e-3 --seed "$seed"
done

# The second point on the size curve, at the same epoch count as the tiny runs
# it is compared against.
for seed in 7 17 27; do
  run "small-synthetic-s$seed" \
    data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
    --preset small --output "artifacts/blink-small-synthetic-s$seed.blink" \
    --epochs 60 --device mps --learning-rate 2e-3 --seed "$seed"
done

echo "[s3] all done $(date +%H:%M:%S)"
