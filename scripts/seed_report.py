#!/usr/bin/env python3
"""Aggregate the multi-seed ablation into one table.

A single run of this model says very little: the plateau transitions are sharp
and a run either makes one or does not. This reports each seed and the spread,
so a difference between the two arms is only claimed when it is larger than the
variation within them.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

SEEDS = (7, 17, 27)
# "cross" is the full model; "film" is it without the question-to-state cross
# layer; "nofilm" drops the query conditioning as well.
ARMS = ("cross", "film", "nofilm")
DATA = ROOT / "data/synthetic/test.jsonl"

# judgment emits three rows per scenario and they are not equally hard. The
# failing mode confuses the first two, which differ only in one verb, so the
# breakdown is what says whether a fix worked.
JUDGMENT_CASES = ("supports", "contradicts", "insufficient")


def evaluate(label: str) -> dict | None:
    report = ROOT / f"results/eval-{label}.json"
    model = ROOT / f"artifacts/blink-{label}.blink"
    if not model.exists():
        return None
    if not report.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "eval/evaluate.py"), str(model), str(DATA),
             "--json", str(report), "--skip-controls", "--bootstrap", "0", "--quiet"],
            check=True, cwd=ROOT,
        )
    return json.loads(report.read_text())


def judgment_breakdown(label: str) -> dict[str, float] | None:
    """Per-sub-case accuracy for the judgment family."""
    from blink_train.data import read_jsonl

    predictions = ROOT / f"results/predictions/{label}.jsonl"
    model = ROOT / f"artifacts/blink-{label}.blink"
    if not model.exists():
        return None
    if not predictions.exists():
        predictions.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(ROOT / "eval/evaluate.py"), str(model), str(DATA),
             "--json", "/dev/null", "--predictions", str(predictions),
             "--skip-controls", "--bootstrap", "0", "--quiet"],
            check=True, cwd=ROOT,
        )
    rows = read_jsonl(DATA)
    hit = {c: [0, 0] for c in JUDGMENT_CASES}
    for line in predictions.read_text().splitlines():
        p = json.loads(line)
        row = rows[p["index"]]
        if row.family != "judgment":
            continue
        text = row.options[row.label]
        case = next(c for c in JUDGMENT_CASES if c in text)
        hit[case][1] += 1
        hit[case][0] += int(p["argmax"] == p["label"])
    return {c: hit[c][0] / hit[c][1] for c in JUDGMENT_CASES if hit[c][1]}


def spread(values: list[float]) -> str:
    if len(values) < 2:
        return f"{values[0]:.4f}" if values else "-"
    return (f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} "
            f"[{min(values):.4f}, {max(values):.4f}]")


def main() -> None:
    reports = {(arm, seed): evaluate(f"tiny-{arm}-s{seed}")
               for arm in ARMS for seed in SEEDS}
    available = {k: v for k, v in reports.items() if v}
    if not available:
        raise SystemExit("no seed runs found; run scripts/queue_seeds.sh first")

    families = sorted(next(iter(available.values()))["by_family"])
    rows = ["pooled"] + families

    print(f"{'slice':12s} {'seed':>5s} {'film':>9s} {'nofilm':>9s} {'delta':>9s}")
    for slice_ in rows:
        for seed in SEEDS:
            got = []
            for arm in ARMS:
                r = reports.get((arm, seed))
                if not r:
                    got.append(None)
                elif slice_ == "pooled":
                    got.append(r["metrics"]["top1"])
                else:
                    got.append(r["by_family"][slice_]["top1"])
            if all(v is not None for v in got):
                print(f"{slice_:12s} {seed:5d} {got[0]:9.4f} {got[1]:9.4f} "
                      f"{got[0] - got[1]:+9.4f}")
        print()

    print(f"{'slice':12s} {'arm':>7s}  mean +- sd [min, max]")
    verdict = {}
    for slice_ in rows:
        values = {}
        for arm in ARMS:
            vals = []
            for seed in SEEDS:
                r = reports.get((arm, seed))
                if not r:
                    continue
                vals.append(r["metrics"]["top1"] if slice_ == "pooled"
                            else r["by_family"][slice_]["top1"])
            values[arm] = vals
            print(f"{slice_:12s} {arm:>7s}  {spread(vals)}")
        a, b = values.get("cross", []), values.get("film", [])
        if len(a) > 1 and len(b) > 1:
            gap = statistics.mean(a) - statistics.mean(b)
            noise = max(statistics.stdev(a), statistics.stdev(b))
            verdict[slice_] = (gap, noise, abs(gap) > 2 * noise)
        del gap, noise
        print()

    print("judgment sub-cases (the failing mode confuses the first two,")
    print("which differ only in one verb):")
    print(f"{'arm':>7s} {'seed':>5s} " + " ".join(f"{c:>13s}" for c in JUDGMENT_CASES))
    for arm in ARMS:
        for seed in SEEDS:
            b = judgment_breakdown(f"tiny-{arm}-s{seed}")
            if b:
                print(f"{arm:>7s} {seed:5d} " +
                      " ".join(f"{b.get(c, float('nan')):13.4f}" for c in JUDGMENT_CASES))
        print()


if __name__ == "__main__":
    main()
