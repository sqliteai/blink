#!/usr/bin/env bash
# jevlike on the same corpus Blink is measured on, at two capacities.
#
# Default width (45k parameters) is what jevlike ships. Width 288 (398k) is
# matched to blink-tiny's 407k, so the comparison is not just a size
# comparison. Sixty epochs and jevlike's own default learning rate, so nothing
# is tuned in Blink's favour.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python"
PY=.venv/bin/python
mkdir -p artifacts/external results

for width in 64 288; do
  for seed in 7 17 27; do
    label="external-w$width-s$seed"
    echo "[jl] $label starting $(date +%H:%M:%S)"
    "$PY" -m jevlike.train data/external/train.jsonl \
      --validation data/external/validation.jsonl \
      --output "artifacts/external/$label.pt" \
      --width "$width" --rank "$width" \
      --context-tokens 256 --option-tokens 48 \
      --epochs 60 --learning-rate 2e-3 --device mps --seed "$seed" \
      > "results/train-$label.jsonl" 2>&1
    echo "[jl] $label exit=$? $(date +%H:%M:%S)"
  done
done
echo "[jl] all done $(date +%H:%M:%S)"
