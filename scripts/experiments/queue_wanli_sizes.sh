#!/usr/bin/env bash
# blink-tiny and blink-small on WANLI, both on the current architecture
# (cosine option head, format 3), so the size comparison changes one variable.
#
# The question is not whether small scores higher -- learning the label prior
# better would do that -- but whether its margin over the shuffled-state
# control grows, i.e. whether capacity buys reading. tiny's margin is about
# five points.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python"
PY=.venv/bin/python

run() {
  local label="$1"; shift
  echo "[wz] $label starting $(date +%H:%M:%S)"
  "$PY" -m blink_train.train "$@" > "results/train-$label.jsonl" 2>&1
  echo "[wz] $label exit=$? $(date +%H:%M:%S)"
}

for seed in 7 17 27; do
  run "tiny-wanli-cos-s$seed" \
    data/wanli/train.jsonl --validation data/wanli/validation.jsonl \
    --preset tiny --limits-from-data \
    --output "artifacts/blink-tiny-wanli-cos-s$seed.blink" \
    --epochs 25 --device mps --learning-rate 3e-3 --seed "$seed"
done

# small left its plateau late on the synthetic corpus, so it gets more than
# tiny's 25 epochs.
for seed in 7 17 27; do
  run "small-wanli-cos-s$seed" \
    data/wanli/train.jsonl --validation data/wanli/validation.jsonl \
    --preset small --limits-from-data \
    --output "artifacts/blink-small-wanli-cos-s$seed.blink" \
    --epochs 60 --device mps --learning-rate 2e-3 --seed "$seed"
done
echo "[wz] all done $(date +%H:%M:%S)"
