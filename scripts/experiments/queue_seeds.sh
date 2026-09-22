#!/usr/bin/env bash
# Three seeds per arm of the FiLM ablation.
#
# A single seed per arm gave a different answer on each of two corpus
# regenerations -- FiLM helping `judgment`, then helping `comparison`, then
# hurting. The plateau transitions in this model are sharp, which is exactly
# the regime where one run says nothing. This settles it, or shows that it
# cannot be settled at this scale.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python"
PY=.venv/bin/python

for seed in 7 17 27; do
  for arm in film nofilm; do
    flag=""
    [ "$arm" = "nofilm" ] && flag="--no-film"
    label="tiny-$arm-s$seed"
    echo "[seeds] $label starting $(date +%H:%M:%S)"
    # shellcheck disable=SC2086
    "$PY" -m blink_train.train \
      data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
      --preset tiny $flag --output "artifacts/blink-$label.blink" \
      --epochs 60 --device mps --learning-rate 3e-3 --seed "$seed" \
      > "results/train-$label.jsonl" 2>&1
    echo "[seeds] $label exit=$? $(date +%H:%M:%S)"
  done
done
echo "[seeds] all done $(date +%H:%M:%S)"
