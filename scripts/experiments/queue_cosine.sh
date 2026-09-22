#!/usr/bin/env bash
# Retrain after the cosine option head and the temperature-bound check.
#
# blink-tiny first: the cosine head changes the architecture, so its old
# numbers no longer describe the model, and a regression there would make the
# longer blink-small runs pointless. blink-small then gets 100 epochs instead
# of 60 -- two of its three runs only left their plateau at epoch 41 and 55,
# which was the other half of why it looked broken.
#
# Waits on the Wikispeedia queue's log rather than on pgrep: a pgrep pattern
# naming the training module also matches this script's own command line.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python"
PY=.venv/bin/python

until grep -q "all done" results/wikispeedia.log 2>/dev/null; do sleep 60; done
echo "[cos] wikispeedia queue finished, starting $(date +%H:%M:%S)"

run() {
  local label="$1"; shift
  echo "[cos] $label starting $(date +%H:%M:%S)"
  "$PY" -m blink_train.train "$@" > "results/train-$label.jsonl" 2>&1
  echo "[cos] $label exit=$? $(date +%H:%M:%S)"
}

for seed in 7 17 27; do
  run "tiny-cosine-s$seed" \
    data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
    --preset tiny --output "artifacts/blink-tiny-cosine-s$seed.blink" \
    --epochs 60 --device mps --learning-rate 3e-3 --seed "$seed"
done

for seed in 7 17 27; do
  run "small-cosine-s$seed" \
    data/synthetic/train.jsonl --validation data/synthetic/validation.jsonl \
    --preset small --output "artifacts/blink-small-cosine-s$seed.blink" \
    --epochs 100 --device mps --learning-rate 2e-3 --seed "$seed"
done

echo "[cos] all done $(date +%H:%M:%S)"
