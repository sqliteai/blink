#!/usr/bin/env python3
"""Render results/*.json as the tables that go in docs/RESULTS.md.

Reads only what a run actually produced, so a partial run yields a partial
summary rather than an error.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(pattern: str) -> dict[str, dict]:
    """Every report this script understands.

    Reports written by other tools -- eval/compare_external.py, for one -- carry
    a different `model` block. They are skipped rather than crashed on, so
    adding a comparison never breaks the summary.
    """
    out = {}
    for path in sorted(RESULTS.glob(pattern)):
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report.get("model"), dict) and "name" in report["model"]:
            out[path.stem] = report
    return out


def read_jsonl(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def human(count: float) -> str:
    for unit in ("B", "KiB", "MiB"):
        if count < 1024 or unit == "MiB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} MiB"


def accuracy_table(reports: dict[str, dict]) -> list[str]:
    lines = [
        "| report | model | rows | top-1 | balanced | macro F1 | ECE | Brier | uniform | clipped |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        metrics = report.get("metrics")
        if not metrics:
            continue
        lines.append(
            f"| {name.removeprefix('eval-')} | {report['model']['name']} "
            f"| {metrics['rows']} | {metrics['top1']:.3f} "
            f"| {metrics['balanced_accuracy']:.3f} | {metrics['macro_f1']:.3f} "
            f"| {metrics['ece']:.3f} | {metrics['brier']:.3f} "
            f"| {metrics['uniform_baseline']:.3f} "
            f"| {report['data'].get('clipped_rows', 0)} |"
        )
    return lines


def family_table(reports: dict[str, dict]) -> list[str]:
    lines = ["| report | family | rows | top-1 | uniform |", "|---|---|---:|---:|---:|"]
    for name, report in reports.items():
        for family, metrics in report.get("by_family", {}).items():
            lines.append(
                f"| {name.removeprefix('eval-')} | {family} | {metrics['rows']} "
                f"| {metrics['top1']:.3f} | {metrics['uniform_baseline']:.3f} |"
            )
    return lines


def control_table(reports: dict[str, dict]) -> list[str]:
    lines = [
        "| report | intact | shuffled state | shuffled question | uniform |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        controls = report.get("controls")
        if not controls:
            continue
        lines.append(
            f"| {name.removeprefix('eval-')} "
            f"| {report['metrics']['top1']:.3f} "
            f"| {controls['shuffled_state']['top1']:.3f} "
            f"| {controls['shuffled_question']['top1']:.3f} "
            f"| {report['metrics']['uniform_baseline']:.3f} |"
        )
    return lines


def perturbation_table(reports: dict[str, dict]) -> list[str]:
    lines = [
        "| report | perturbation | argmax flip rate | mean total variation | top-1 before | after |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        for kind, metrics in report.get("perturbations", {}).items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                f"| {name.removeprefix('eval-')} | {kind} "
                f"| {metrics['argmax_flip_rate']:.3f} "
                f"| {metrics['mean_total_variation']:.3f} "
                f"| {metrics['accuracy_before']:.3f} "
                f"| {metrics['accuracy_after']:.3f} |"
            )
    return lines


ABLATION_PAIRS = (
    ("tiny-synthetic", "tiny-synthetic-nofilm",
     "blink-tiny on the full synthetic corpus"),
    ("tiny-selection", "tiny-selection-nofilm",
     "blink-tiny on the selection family alone"),
)


def ablation_table(reports: dict[str, dict]) -> list[str]:
    """Side by side, with and without the question conditioning.

    The pooled row is reported for completeness; the per-family rows are the
    result, because the families that need a question-by-option interaction are
    the only ones the ablation can affect.
    """
    lines: list[str] = []
    for with_film, without, label in ABLATION_PAIRS:
        a = reports.get(f"eval-{with_film}")
        b = reports.get(f"eval-{without}")
        if not a or not b:
            continue
        lines += [
            f"**{label}**", "",
            "| slice | with FiLM | without FiLM | uniform | rows |",
            "|---|---:|---:|---:|---:|",
            f"| all | {a['metrics']['top1']:.3f} | {b['metrics']['top1']:.3f} "
            f"| {a['metrics']['uniform_baseline']:.3f} | {a['metrics']['rows']} |",
        ]
        for family in sorted(set(a.get("by_family", {})) & set(b.get("by_family", {}))):
            fa, fb = a["by_family"][family], b["by_family"][family]
            lines.append(
                f"| {family} | {fa['top1']:.3f} | {fb['top1']:.3f} "
                f"| {fa['uniform_baseline']:.3f} | {fa['rows']} |"
            )
        lines.append("")
    return lines


def latency_table(rows: list[dict]) -> list[str]:
    lines = [
        "| model | path | state bytes | options | p50 us | p90 us | p99 us | decisions/s | arena |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['path']} | {row['state_bytes']} "
            f"| {row['options']} | {row['p50_us']:.1f} | {row['p90_us']:.1f} "
            f"| {row['p99_us']:.1f} | {row['decisions_per_second']:.0f} "
            f"| {human(row['arena_bytes'])} |"
        )
    return lines


def memory_table(rows: list[dict]) -> list[str]:
    lines = [
        "| model | parameters | weights | bytes/param | minimum arena | default arena |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        arenas = row.get("arenas")
        if not arenas:
            continue
        lines.append(
            f"| {row['model']} | {row['parameters']:,} "
            f"| {human(row['weights_bytes'])} "
            f"| {row['bytes_per_parameter']:.2f} "
            f"| {human(arenas['minimum'])} | {human(arenas['default'])} |"
        )
    return lines


def main() -> None:
    reports = load("eval-*.json")
    out: list[str] = ["# Results (generated by scripts/summarise.py)", ""]

    if reports:
        out += ["## Accuracy", ""] + accuracy_table(reports) + [""]
        out += ["## By task family", ""] + family_table(reports) + [""]
        controls = control_table(reports)
        if len(controls) > 2:
            out += ["## Controls", "",
                    "A model that scores well with a shuffled state is reading "
                    "the option prior, not the evidence.", ""] + controls + [""]
        perturbations = perturbation_table(reports)
        if len(perturbations) > 2:
            out += ["## Perturbations", ""] + perturbations + [""]
        ablation = ablation_table(reports)
        if ablation:
            out += [
                "## Ablation: question conditioning of the option queries", "",
                "Identical settings and seed; only the FiLM conditioning "
                "differs. Without it an option's query is a function of the "
                "option text alone, so the score has no question-by-option "
                "term at all.", "",
            ] + ablation

    for label, name in (("Latency (synthesised weights)", "bench-latency.jsonl"),
                        ("Latency (trained weights)", "bench-latency-trained.jsonl")):
        rows = [row for row in read_jsonl(name) if "p50_us" in row]
        if rows:
            out += [f"## {label}", ""] + latency_table(rows) + [""]

    for label, name in (("Memory (synthesised weights)", "bench-memory.jsonl"),
                        ("Memory (trained weights)", "bench-memory-trained.jsonl")):
        rows = [row for row in read_jsonl(name) if "arenas" in row]
        if rows:
            out += [f"## {label}", ""] + memory_table(rows) + [""]

    print("\n".join(out))


if __name__ == "__main__":
    main()
