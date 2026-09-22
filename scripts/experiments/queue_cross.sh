#!/usr/bin/env bash
# Three seeds with the question-to-state cross layer, against the three
# already run without it.
#
# `judgment` was bimodal: half the runs learned to match the verb in the
# question against the verb in the state and half did not. Nothing in the
# encoder let those two positions meet, so the transition had to be found
# through the option head's weighted average. This measures whether giving the
# question a direct read of the state removes the coin flip.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python"
PY=.venv/bin/python

for seed in 7 17 27; do
  label="tiny-cross-s$seed"
  echo "[cross] $label starting $(date +%H:%M:%S)"
  "$PY" -m blink_train.train \
    data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
    --preset tiny --output "artifacts/blink-$label.blink" \
    --epochs 60 --device mps --learning-rate 3e-3 --seed "$seed" \
    > "results/train-$label.jsonl" 2>&1
  echo "[cross] $label exit=$? $(date +%H:%M:%S)"
done
echo "[cross] all done $(date +%H:%M:%S)"
