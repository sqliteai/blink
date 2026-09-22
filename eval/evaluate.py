#!/usr/bin/env python3
"""Accuracy, calibration and control harness for a .blink model.

Every number this prints comes from the C runtime through the ctypes binding,
so it describes the artefact that ships rather than the PyTorch model it was
trained from.

What it reports, and why each one is here:

  top-1 / top-3          the headline, with the full denominator
  balanced accuracy      the headline again, immune to a skewed label prior
  macro F1               per-class behaviour, not just the majority class
  NLL / Brier            proper scoring rules; they punish confident mistakes
  ECE / MCE              calibration: does 0.8 confidence mean 80% correct
  risk-coverage          accuracy when the model is allowed to abstain below a
                         confidence threshold, which is how a typed decision
                         actually gets used in software

Controls, which a reported accuracy is meaningless without:

  shuffled state         each menu paired with another row's state
  shuffled question      each state paired with another row's question
  uniform baseline       1/N averaged over rows, the real chance level when
                         option counts vary
  majority baseline      always answer the most common label index

Bootstrap intervals resample groups, not rows, because several questions can
share one state.

Usage:
    python eval/evaluate.py MODEL.blink DATA.jsonl [--json OUT] [--predictions OUT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from blink_train.data import Row, clip, read_jsonl  # noqa: E402
from blink_train.runtime import Model  # noqa: E402

CHANCE_EPSILON = 1e-12


# --------------------------------------------------------------- prediction


def score_rows(session, rows: list[Row], question_override=None,
               state_override=None) -> list[dict]:
    """Run every row through the C runtime, reusing the cached state when two
    consecutive rows share one. That reuse is exact, not an approximation."""
    predictions = []
    cached_state = None
    for index, row in enumerate(rows):
        state = row.state if state_override is None else state_override[index]
        question = row.question if question_override is None else question_override[index]
        if state != cached_state:
            session.set_state(clip(state, session.model.info.max_state))
            cached_state = state
        session.set_menu([clip(option, session.model.info.max_option) or b" "
                          for option in row.options])
        started = time.perf_counter()
        probabilities, result = session.score(clip(question, session.model.info.max_question))
        elapsed = time.perf_counter() - started
        predictions.append({
            "index": index,
            "group": row.group,
            "family": row.family,
            "label": row.label,
            "options": len(row.options),
            "probabilities": [float(p) for p in probabilities],
            "argmax": int(result.argmax),
            "confidence": float(result.confidence),
            "entropy": float(result.entropy),
            "margin": float(result.margin),
            "seconds": elapsed,
        })
    return predictions


# ------------------------------------------------------------------ metrics


def _safe_log(value: float) -> float:
    return math.log(max(value, CHANCE_EPSILON))


def expected_calibration_error(predictions: list[dict], bins: int = 15) -> dict:
    """Equal-width binning of the top-1 confidence against its accuracy."""
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for item in predictions:
        index = min(bins - 1, int(item["confidence"] * bins))
        buckets[index].append((item["confidence"], int(item["argmax"] == item["label"])))

    total = len(predictions)
    ece = 0.0
    mce = 0.0
    table = []
    for index in sorted(buckets):
        entries = buckets[index]
        confidence = sum(c for c, _ in entries) / len(entries)
        accuracy = sum(h for _, h in entries) / len(entries)
        gap = abs(confidence - accuracy)
        ece += len(entries) / total * gap
        mce = max(mce, gap)
        table.append({
            "bin": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": len(entries),
            "mean_confidence": confidence,
            "accuracy": accuracy,
        })
    return {"ece": ece, "mce": mce, "bins": table}


def risk_coverage(predictions: list[dict]) -> list[dict]:
    """Accuracy at several abstention thresholds. A typed decision is only
    useful if refusing the uncertain cases actually buys accuracy."""
    out = []
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        kept = [p for p in predictions if p["confidence"] >= threshold]
        if not kept:
            out.append({"threshold": threshold, "coverage": 0.0, "accuracy": None})
            continue
        correct = sum(p["argmax"] == p["label"] for p in kept)
        out.append({
            "threshold": threshold,
            "coverage": len(kept) / len(predictions),
            "accuracy": correct / len(kept),
        })
    return out


def core_metrics(predictions: list[dict]) -> dict:
    total = len(predictions)
    correct = sum(p["argmax"] == p["label"] for p in predictions)

    top3 = 0
    nll = 0.0
    brier = 0.0
    for item in predictions:
        ranking = sorted(range(item["options"]),
                         key=lambda i: item["probabilities"][i], reverse=True)
        if item["label"] in ranking[:3]:
            top3 += 1
        nll -= _safe_log(item["probabilities"][item["label"]])
        for index in range(item["options"]):
            target = 1.0 if index == item["label"] else 0.0
            brier += (item["probabilities"][index] - target) ** 2

    # Balanced accuracy and macro F1 over label indices actually present.
    per_class: dict[int, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
    )
    for item in predictions:
        per_class[item["label"]]["support"] += 1
        if item["argmax"] == item["label"]:
            per_class[item["label"]]["tp"] += 1
        else:
            per_class[item["label"]]["fn"] += 1
            per_class[item["argmax"]]["fp"] += 1

    recalls, f1s = [], []
    for stats in per_class.values():
        if stats["support"]:
            recalls.append(stats["tp"] / stats["support"])
        denominator = 2 * stats["tp"] + stats["fp"] + stats["fn"]
        f1s.append(2 * stats["tp"] / denominator if denominator else 0.0)

    uniform = sum(1.0 / p["options"] for p in predictions) / total
    majority_label = Counter(p["label"] for p in predictions).most_common(1)[0][0]
    majority = sum(p["label"] == majority_label for p in predictions) / total

    calibration = expected_calibration_error(predictions)
    return {
        "rows": total,
        "top1": correct / total,
        "top3": top3 / total,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "nll": nll / total,
        "brier": brier / total,
        "ece": calibration["ece"],
        "mce": calibration["mce"],
        "mean_confidence": sum(p["confidence"] for p in predictions) / total,
        "mean_entropy": sum(p["entropy"] for p in predictions) / total,
        "uniform_baseline": uniform,
        "majority_baseline": majority,
        "calibration_bins": calibration["bins"],
        "risk_coverage": risk_coverage(predictions),
        "median_seconds": sorted(p["seconds"] for p in predictions)[total // 2],
    }


def bootstrap_interval(predictions: list[dict], metric: str, seed: int = 20260920,
                       resamples: int = 2000) -> dict:
    """Percentile bootstrap over groups. Rows that share a state move together,
    otherwise the interval would be optimistically narrow."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for item in predictions:
        by_group[item["group"] or f"row-{item['index']}"].append(item)
    groups = list(by_group.values())

    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        drawn = []
        for _ in range(len(groups)):
            drawn.extend(rng.choice(groups))
        samples.append(core_metrics(drawn)[metric])
    samples.sort()
    return {
        "metric": metric,
        "groups": len(groups),
        "resamples": resamples,
        "low": samples[int(0.025 * resamples)],
        "high": samples[int(0.975 * resamples)],
    }


def by_family(predictions: list[dict]) -> dict:
    """Never pool distinct task families into one accuracy number."""
    families: dict[str, list[dict]] = defaultdict(list)
    for item in predictions:
        families[item["family"] or "unlabelled"].append(item)
    return {
        name: {
            key: value
            for key, value in core_metrics(items).items()
            if key not in ("calibration_bins", "risk_coverage")
        }
        for name, items in sorted(families.items())
    }


# ----------------------------------------------------------------- controls


def shuffled(values: list, seed: int) -> list:
    """A cyclic derangement: every position receives a different position's
    value. A plain shuffle would leave some rows paired with themselves, and
    those rows would silently contribute nothing to the control.

    Positions are deranged, not values: if two rows happen to carry identical
    text, one can still receive its own text, and there is nothing a shuffle
    can do about that.
    """
    count = len(values)
    if count < 2:
        return list(values)
    rng = random.Random(seed)
    order = list(range(count))
    rng.shuffle(order)
    out = [None] * count
    for position, index in enumerate(order):
        out[index] = values[order[(position + 1) % count]]
    return out


def run_controls(session, rows: list[Row], seed: int) -> dict:
    states = [row.state for row in rows]
    questions = [row.question for row in rows]
    summary = {}

    for name, kwargs in (
        ("shuffled_state", {"state_override": shuffled(states, seed)}),
        ("shuffled_question", {"question_override": shuffled(questions, seed + 1)}),
    ):
        predictions = score_rows(session, rows, **kwargs)
        summary[name] = {
            key: value
            for key, value in core_metrics(predictions).items()
            if key not in ("calibration_bins", "risk_coverage")
        }
    return summary


def run_perturbations(session, rows: list[Row]) -> dict:
    """Output-blind perturbations that must not change a decision's meaning.

    Option order is reversed and the label index moved with it, so a model that
    tracks the text rather than the position should be unmoved. Irrelevant
    trailing context is appended to the state.
    """
    baseline = score_rows(session, rows)

    reversed_rows = [
        Row(row.state, row.question, tuple(reversed(row.options)),
            len(row.options) - 1 - row.label, row.group, row.family)
        for row in rows
    ]
    reversed_predictions = score_rows(session, reversed_rows)

    filler = " The office will be closed for maintenance next month."
    padded_rows = [
        Row(row.state + filler, row.question, row.options, row.label,
            row.group, row.family)
        for row in rows
    ]
    padded_predictions = score_rows(session, padded_rows)

    def compare(before: list[dict], after: list[dict], remap) -> dict:
        flips = 0
        movement = 0.0
        for b, a in zip(before, after):
            if remap(b["argmax"], b["options"]) != a["argmax"]:
                flips += 1
            aligned = [a["probabilities"][remap(i, b["options"])]
                       for i in range(b["options"])]
            movement += sum(abs(x - y) for x, y in zip(b["probabilities"], aligned))
        return {
            "rows": len(before),
            "argmax_flip_rate": flips / len(before),
            "mean_total_variation": movement / (2 * len(before)),
            "accuracy_before": core_metrics(before)["top1"],
            "accuracy_after": core_metrics(after)["top1"],
        }

    return {
        "reversed_options": compare(baseline, reversed_predictions,
                                    lambda i, n: n - 1 - i),
        "irrelevant_suffix": compare(baseline, padded_predictions,
                                     lambda i, _n: i),
        "note": "The suffix is appended and the harness clips to the model's "
                "byte limit, so on a state already at the limit the "
                "perturbation degrades to a no-op rather than to a truncation "
                "of the original evidence.",
    }


# --------------------------------------------------------------------- main


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: str | Path) -> str:
    """A path as it should appear in a published report: relative to the
    repository when it is inside it, so reports do not carry the home
    directory of whoever ran them."""
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("data")
    parser.add_argument("--json", help="write the full report here")
    parser.add_argument("--predictions", help="write row-level predictions here")
    parser.add_argument("--max-options", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    rows = read_jsonl(args.data)
    model = Model(args.model)
    info = model.info
    session = model.session(max_options=max(2, min(args.max_options,
                                                   max(len(r.options) for r in rows))))

    clipped = sum(
        1 for row in rows
        if len(row.state.encode()) > info.max_state
        or len(row.question.encode()) > info.max_question
        or any(len(o.encode()) > info.max_option for o in row.options)
    )

    predictions = score_rows(session, rows)
    report = {
        "model": {
            "path": portable(args.model),
            "sha256": file_digest(Path(args.model)),
            "name": info.name.decode(),
            "parameters": int(info.parameters),
            "weights_bytes": int(info.weights_bytes),
            "temperature": float(info.temperature),
            "max_state": int(info.max_state),
            "max_question": int(info.max_question),
            "max_option": int(info.max_option),
        },
        "data": {
            "path": portable(args.data),
            "sha256": file_digest(Path(args.data)),
            "rows": len(rows),
            "groups": len({row.group for row in rows}),
            "clipped_rows": clipped,
            "clipping_note": "inputs longer than the model limits are truncated "
                             "by the harness; the runtime itself refuses them",
        },
        "session_arena_bytes": session.arena_bytes,
        "metrics": core_metrics(predictions),
        "by_family": by_family(predictions),
    }
    if args.bootstrap:
        report["intervals"] = [
            bootstrap_interval(predictions, metric, args.seed, args.bootstrap)
            for metric in ("top1", "balanced_accuracy")
        ]
    if not args.skip_controls:
        report["controls"] = run_controls(session, rows, args.seed)
        report["perturbations"] = run_perturbations(session, rows)

    if args.predictions:
        path = Path(args.predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item) + "\n")
        report["predictions_path"] = portable(path)
        report["predictions_sha256"] = file_digest(path)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        metrics = report["metrics"]
        print(f"model   {report['model']['name']}  "
              f"{report['model']['weights_bytes'] / 1024:.0f} KiB weights, "
              f"{report['session_arena_bytes'] / 1024:.0f} KiB arena")
        print(f"data    {args.data}  {metrics['rows']} rows, "
              f"{report['data']['groups']} groups, {clipped} clipped")
        print(f"top1    {metrics['top1']:.4f}   "
              f"balanced {metrics['balanced_accuracy']:.4f}   "
              f"macroF1 {metrics['macro_f1']:.4f}")
        print(f"cal     ece {metrics['ece']:.4f}   nll {metrics['nll']:.4f}   "
              f"brier {metrics['brier']:.4f}")
        print(f"chance  uniform {metrics['uniform_baseline']:.4f}   "
              f"majority {metrics['majority_baseline']:.4f}")
        if "controls" in report:
            for name, control in report["controls"].items():
                print(f"control {name:<18} top1 {control['top1']:.4f}")
        for name, family in report["by_family"].items():
            print(f"family  {name:<18} top1 {family['top1']:.4f} "
                  f"({family['rows']} rows, uniform {family['uniform_baseline']:.3f})")

    session.close()
    model.close()
    return report


if __name__ == "__main__":
    main()
