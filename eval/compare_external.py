#!/usr/bin/env python3
"""Head-to-head against jevlike on one corpus, scored by one harness.

jevlike is the fair comparison: both models are small and both are trained
from scratch. Its frozen-encoder results are a different experiment -- they
measure a pretrained encoder, not the scorer.

Two things make a comparison of this kind dishonest by default, and both are
handled here:

* **Truncation.** jevlike takes one `context` and no question, and its default
  is 192 bytes. Folding Blink's question into that context and then letting it
  truncate would be measuring the truncation. The converter records the longest
  row and `--context-tokens` is set from it.
* **Capacity.** jevlike at its defaults is about a tenth of blink-tiny. Both
  sizes are reported: its default, and a width matched to Blink's parameter
  count.

Predictions from both models go through `eval/evaluate.py`'s metric code, so
the numbers are computed by identical arithmetic.

    python eval/compare_external.py convert --output data/external
    python eval/compare_external.py score CHECKPOINT.pt --data data/external/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blink_train.data import Row, read_jsonl  # noqa: E402

SEPARATOR = "\n"


def to_external(row: Row) -> dict:
    """One Blink row as a jevlike row.

    jevlike has no question field, so the question is prefixed to the state.
    It goes first because it is short and because a model that truncates
    should lose the tail of the evidence rather than the criterion -- the
    kinder of the two failure modes for jevlike, which is the point.
    """
    context = f"{row.question}{SEPARATOR}{row.state}" if row.question else row.state
    return {"context": context, "options": list(row.options), "label": row.label}


def convert(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"separator": repr(SEPARATOR), "splits": {}}
    longest_context = longest_option = 0

    for split in ("train", "validation", "test"):
        source = args.source / f"{split}.jsonl"
        if not source.exists():
            continue
        rows = read_jsonl(source)
        converted = [to_external(row) for row in rows]
        target = args.output / f"{split}.jsonl"
        target.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in converted) + "\n",
            encoding="utf-8",
        )
        context_bytes = max(len(r["context"].encode()) for r in converted)
        option_bytes = max(len(o.encode()) for r in converted for o in r["options"])
        longest_context = max(longest_context, context_bytes)
        longest_option = max(longest_option, option_bytes)
        report["splits"][split] = {
            "rows": len(converted),
            "max_context_bytes": context_bytes,
            "max_option_bytes": option_bytes,
        }

    # Round up so nothing is clipped. A comparison against a truncated model
    # measures the truncation.
    report["recommended"] = {
        "context_tokens": ((longest_context + 31) // 32) * 32,
        "option_tokens": ((longest_option + 15) // 16) * 16,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def native_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def split_header(context: str, header_lines: int) -> tuple[str, str]:
    """Separate the navigation header from the evidence in a native row."""
    parts = context.split("\n", header_lines)
    if len(parts) <= header_lines:
        return context, ""
    return "\n".join(parts[:header_lines]), parts[header_lines]


def controlled(rows: list[dict], control: str, header_lines: int,
               seed: int) -> list[dict]:
    """Rows with the evidence or the header deranged across rows.

    jevlike's own shuffle_context control swaps whole contexts, titles
    included, which answers a different question. This swaps only the part
    named, so it matches the controls Blink is measured with: `state` keeps
    each row's navigation header and gives it another row's article body;
    `question` does the reverse.
    """
    if control == "none":
        return rows
    import evaluate

    headers, bodies = zip(*(split_header(r["context"], header_lines) for r in rows))
    if control == "state":
        bodies = evaluate.shuffled(list(bodies), seed)
    elif control == "question":
        headers = evaluate.shuffled(list(headers), seed)
    else:
        raise ValueError(control)
    return [{**r, "context": f"{h}\n{b}" if b else h}
            for r, h, b in zip(rows, headers, bodies)]


def score_native(args: argparse.Namespace) -> None:
    """Score a jevlike checkpoint on rows already in jevlike's own format,
    under each control, so the comparison with Blink's controls is exact."""
    import torch

    from jevlike.data import ChoiceExample
    from jevlike.model import load_checkpoint, select_device
    from jevlike.train import move

    import evaluate

    device = select_device(args.device)
    model, collator, config = load_checkpoint(args.checkpoint, device)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base = native_rows(args.data)

    report = {"model": {"path": str(args.checkpoint), "parameters": parameters,
                        "config": config, "system": "jevlike"},
              "data": {"path": str(args.data), "rows": len(base)},
              "controls": {}}

    for control in ("none", "state", "question"):
        rows = controlled(base, control, args.header_lines, args.seed)
        predictions = []
        for index, row in enumerate(rows):
            example = ChoiceExample(row["context"], tuple(row["options"]), row["label"])
            batch = move(collator([example]), device)
            with torch.no_grad():
                probabilities = (model(batch).softmax(-1)[0, :len(row["options"])]
                                 .cpu().tolist())
            argmax = max(range(len(probabilities)), key=probabilities.__getitem__)
            predictions.append({
                "index": index, "group": "", "family": "native",
                "label": row["label"], "options": len(row["options"]),
                "probabilities": probabilities, "argmax": argmax,
                "confidence": float(probabilities[argmax]),
                "entropy": 0.0, "margin": 0.0, "seconds": 0.0,
            })
        metrics = evaluate.core_metrics(predictions)
        if control == "none":
            report["metrics"] = metrics
        else:
            report["controls"][f"shuffled_{control}"] = {
                k: v for k, v in metrics.items()
                if k not in ("calibration_bins", "risk_coverage")}
        print(f"  {control:9s} top1 {metrics['top1']:.4f}  "
              f"uniform {metrics['uniform_baseline']:.4f}", flush=True)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


def score(args: argparse.Namespace) -> None:
    """Run a jevlike checkpoint and write predictions in Blink's format."""
    import torch

    from jevlike.data import ChoiceExample
    from jevlike.model import load_checkpoint, select_device
    from jevlike.train import move

    rows = read_jsonl(args.data)
    device = select_device(args.device)
    model, collator, config = load_checkpoint(args.checkpoint, device)
    model.eval()

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    predictions = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        example = ChoiceExample(to_external(row)["context"], row.options, row.label)
        batch = move(collator([example]), device)
        with torch.no_grad():
            probabilities = model(batch).softmax(-1)[0, :len(row.options)].cpu().tolist()
        argmax = max(range(len(probabilities)), key=probabilities.__getitem__)
        predictions.append({
            "index": index, "group": row.group, "family": row.family,
            "label": row.label, "options": len(row.options),
            "probabilities": [float(p) for p in probabilities],
            "argmax": argmax, "confidence": float(probabilities[argmax]),
            "entropy": 0.0, "margin": 0.0, "seconds": 0.0,
        })
    elapsed = time.perf_counter() - started

    import evaluate

    metrics = evaluate.core_metrics(predictions)
    report = {
        "model": {"path": str(args.checkpoint), "parameters": parameters,
                  "config": config, "system": "jevlike"},
        "data": {"path": str(args.data), "rows": len(rows), "clipped_rows": 0,
                 "clipping_note": "jevlike truncates internally at "
                                  f"context_tokens={config.get('context_tokens')}"},
        "metrics": metrics,
        "by_family": evaluate.by_family(predictions),
        "seconds_total": elapsed,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    if args.predictions:
        Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
        Path(args.predictions).write_text(
            "\n".join(json.dumps(p) for p in predictions) + "\n")

    print(f"jevlike  {parameters:,} trainable parameters")
    print(f"  top1 {metrics['top1']:.4f}  balanced {metrics['balanced_accuracy']:.4f} "
          f"  ece {metrics['ece']:.4f}  uniform {metrics['uniform_baseline']:.4f}")
    for family, value in report["by_family"].items():
        print(f"  {family:12s} {value['top1']:.4f}  ({value['rows']} rows, "
              f"uniform {value['uniform_baseline']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    c = commands.add_parser("convert", help="Blink rows to jevlike rows")
    c.add_argument("--source", type=Path, default=Path("data/synthetic"))
    c.add_argument("--output", type=Path, default=Path("data/external"))
    c.set_defaults(func=convert)

    s = commands.add_parser("score", help="score a jevlike checkpoint")
    s.add_argument("checkpoint")
    s.add_argument("--data", required=True)
    s.add_argument("--json")
    s.add_argument("--predictions")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=score)

    n = commands.add_parser(
        "score-native",
        help="score on jevlike-format rows, with Blink-style controls")
    n.add_argument("checkpoint")
    n.add_argument("--data", required=True)
    n.add_argument("--json")
    n.add_argument("--header-lines", type=int, default=2,
                   help="lines of navigation header before the evidence")
    n.add_argument("--seed", type=int, default=20260920)
    n.add_argument("--device", default="auto")
    n.set_defaults(func=score_native)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
