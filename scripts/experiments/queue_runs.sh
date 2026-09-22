#!/usr/bin/env bash
# Run the training jobs one after another so they do not contend for the GPU.
# Ordered by how much each one is needed: the reference model and the ablation
# carry the central architectural claim, so they go first.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python"
PY=.venv/bin/python

run() {
  local label="$1"; shift
  echo "[queue] $label starting $(date +%H:%M:%S)"
  "$PY" -m blink_train.train "$@" > "results/train-$label.jsonl" 2>&1
  echo "[queue] $label exit=$? $(date +%H:%M:%S)"
}

run tiny-synthetic \
  data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
  --preset tiny --output artifacts/blink-tiny-synthetic.blink \
  --checkpoint artifacts/blink-tiny-synthetic.pt \
  --epochs 100 --device mps --learning-rate 3e-3 --seed 7

run tiny-synthetic-nofilm \
  data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
  --preset tiny --no-film --output artifacts/blink-tiny-synthetic-nofilm.blink \
  --epochs 100 --device mps --learning-rate 3e-3 --seed 7

# The single-family runs separate "cannot learn this" from "did not learn this
# while three quarters of the gradient came from tasks already solved".
run selection-film \
  data/selection/train.jsonl --validation data/selection/validation.jsonl \
  --preset tiny --output artifacts/blink-tiny-selection.blink \
  --epochs 100 --device mps --learning-rate 3e-3 --seed 7

run selection-nofilm \
  data/selection/train.jsonl --validation data/selection/validation.jsonl \
  --preset tiny --no-film --output artifacts/blink-tiny-selection-nofilm.blink \
  --epochs 100 --device mps --learning-rate 3e-3 --seed 7

# WANLI premises reach 551 bytes against blink-tiny's 256, which truncated
# every row of the previous run. --limits-from-data sizes the container for the
# corpus and records what it chose.
run tiny-wanli \
  data/wanli/train.jsonl --validation data/wanli/validation.jsonl \
  --preset tiny --limits-from-data \
  --output artifacts/blink-tiny-wanli.blink \
  --epochs 25 --device mps --learning-rate 3e-3 --seed 7

run small-synthetic \
  data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
  --preset small --output artifacts/blink-small-synthetic.blink \
  --epochs 40 --device mps --learning-rate 2e-3 --seed 7

echo "[queue] all done $(date +%H:%M:%S)"
