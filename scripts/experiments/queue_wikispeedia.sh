#!/usr/bin/env bash
# Blink and jevlike on Wikispeedia next-click, the task jevlike publishes on.
#
# Same rows, same splits, same menus, same body budget, same epochs. jevlike's
# context length is taken from the data so nothing it reads is truncated;
# Blink's byte limits likewise, via --limits-from-data. Both are scored by
# eval/evaluate.py's metric code.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python"
PY=.venv/bin/python
mkdir -p artifacts/external results
EPOCHS=30

for seed in 7 17 27; do
  label="tiny-wikispeedia-s$seed"
  echo "[ws] $label starting $(date +%H:%M:%S)"
  "$PY" -m blink_train.train \
    data/wikispeedia/blink/train.jsonl \
    --validation data/wikispeedia/blink/validation.jsonl \
    --preset tiny --limits-from-data --max-options 64 \
    --output "artifacts/blink-$label.blink" \
    --epochs "$EPOCHS" --batch-size 32 --device mps \
    --learning-rate 3e-3 --seed "$seed" \
    > "results/train-$label.jsonl" 2>&1
  echo "[ws] $label exit=$? $(date +%H:%M:%S)"
done

for seed in 7 17 27; do
  label="external-ws-s$seed"
  echo "[ws] $label starting $(date +%H:%M:%S)"
  "$PY" -m jevlike.train data/wikispeedia/external/train.jsonl \
    --validation data/wikispeedia/external/validation.jsonl \
    --output "artifacts/external/$label.pt" \
    --width 288 --rank 288 \
    --context-tokens 704 --option-tokens 96 \
    --epochs "$EPOCHS" --batch-size 32 --learning-rate 2e-3 \
    --device mps --seed "$seed" \
    > "results/train-$label.jsonl" 2>&1
  echo "[ws] $label exit=$? $(date +%H:%M:%S)"
done

echo "[ws] all done $(date +%H:%M:%S)"
