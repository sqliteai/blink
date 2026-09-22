#!/usr/bin/env bash
# Distillation from MiniCPM5-2B on WANLI (branch blink-distilled).
#
# Two arms against the gold-only runs already in results/ (tiny-wanli-s*,
# small-wanli-s*, from scripts/run_all.sh with the same presets, epochs,
# learning rates and --limits-from-data): the teacher's distribution alone
# (--distill-alpha 1) and half teacher, half label (0.5). Only the training
# targets change. Checkpoint selection, the temperature fit and every metric
# still use the gold labels, and the test split is the gold one.
#
# Needs the labelled corpus first:
#   .venv-teacher/bin/python scripts/teacher_label.py \
#       artifacts/external/teacher/MiniCPM5-2B data/wanli \
#       --splits validation test train --output data/wanli-minicpm5 \
#       --preamble scripts/teacher_prompts/nli.txt --shots 2 --prior-correction
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python"
PY=.venv/bin/python
CORPUS=data/wanli-minicpm5

until [ -f "$CORPUS/manifest.json" ]; do sleep 60; done

run() {  # label preset epochs lr alpha seed
  local label="$1" preset="$2" epochs="$3" lr="$4" alpha="$5" seed="$6"
  echo "[kd] $label starting $(date +%H:%M:%S)"
  "$PY" -m blink_train.train "$CORPUS/train.jsonl" \
    --validation "$CORPUS/validation.jsonl" \
    --preset "$preset" --limits-from-data --epochs "$epochs" \
    --learning-rate "$lr" --device mps --seed "$seed" \
    --distill-alpha "$alpha" \
    --output "artifacts/blink-$label.blink" \
    --checkpoint "artifacts/blink-$label.pt" \
    > "results/train-$label.jsonl" 2>&1
  echo "[kd] $label exit=$? $(date +%H:%M:%S)"
  "$PY" eval/evaluate.py "artifacts/blink-$label.blink" data/wanli/test.jsonl \
    --quiet --json "results/eval-$label.json" \
    --predictions "results/predictions/$label.jsonl"
  if [ -f data/wanli256/test.jsonl ]; then
    local subset="${label/-wanli-/-wanli256-}"
    "$PY" eval/evaluate.py "artifacts/blink-$label.blink" data/wanli256/test.jsonl \
      --quiet --json "results/eval-$subset.json" \
      --predictions "results/predictions/$subset.jsonl"
  fi
}

for arm in kd100:1.0 kd50:0.5; do
  name="${arm%%:*}" alpha="${arm##*:}"
  for seed in 7 17 27; do
    run "tiny-wanli-$name-s$seed" tiny 25 3e-3 "$alpha" "$seed"
  done
done
for arm in kd100:1.0 kd50:0.5; do
  name="${arm%%:*}" alpha="${arm##*:}"
  for seed in 7 17 27; do
    run "small-wanli-$name-s$seed" small 60 2e-3 "$alpha" "$seed"
  done
done
echo "[kd] all done $(date +%H:%M:%S)"
