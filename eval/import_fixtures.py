#!/usr/bin/env python3
"""Convert the SemIf benchmark fixtures into Blink decision rows.

SemIf (MIT, https://github.com/TheoLeeCJ/SemIf) commits two owned fixtures:

    authored144.jsonl        144 authored decisions, three options each
    perturbations108.jsonl   108 output-blind variants of 36 of them

They map onto Blink's row format one to one, which makes them a useful
external check: the same rows that SemIf scores with a frozen 4B model can be
scored by the C runtime here.

Read the numbers carefully. These fixtures carry no training split, so a Blink
model trained on another corpus is being asked to transfer, not to fit. The
`--split-groups` option instead carves a train/test split out of the fixture
itself, grouped by `group_id` so a perturbation never lands opposite its
original.

    python eval/import_fixtures.py --source ../SemIf --output data/fixtures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from blink_train.data import Row, write_jsonl  # noqa: E402

FIXTURES = ("authored144", "perturbations108")


def convert(payload: dict) -> Row:
    state = payload["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    return Row(
        state=state,
        question=payload["question"],
        options=tuple(option["description"] for option in payload["options"]),
        label=int(payload["label"]),
        group=str(payload.get("group_id", payload.get("id", ""))),
        family=str(payload.get("family", "")),
    )


def load(path: Path) -> list[Row]:
    return [convert(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_by_group(rows: list[Row], fractions: tuple[float, float, float],
                   seed: int) -> dict[str, list[Row]]:
    groups = sorted({row.group for row in rows})
    random.Random(seed).shuffle(groups)
    train_cut = int(len(groups) * fractions[0])
    validation_cut = train_cut + int(len(groups) * fractions[1])
    assignment = {}
    for index, group in enumerate(groups):
        assignment[group] = ("train" if index < train_cut
                             else "validation" if index < validation_cut
                             else "test")
    out: dict[str, list[Row]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        out[assignment[row.group]].append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("../SemIf"),
                        help="a SemIf checkout")
    parser.add_argument("--output", type=Path, default=Path("data/fixtures"))
    parser.add_argument("--split-groups", action="store_true",
                        help="also write a grouped train/validation/test split")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    fixtures = args.source / "benchmarks" / "data"
    if not fixtures.is_dir():
        raise SystemExit(
            f"{fixtures} not found. Clone SemIf, or pass --source.\n"
            "  git clone https://github.com/TheoLeeCJ/SemIf"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "SemIf benchmarks/data (MIT)",
        "upstream": "https://github.com/TheoLeeCJ/SemIf",
        "note": "Converted, not modified: state, question, option descriptions "
                "and labels are copied verbatim.",
        "fixtures": {},
    }

    everything: list[Row] = []
    for name in FIXTURES:
        source = fixtures / f"{name}.jsonl"
        if not source.exists():
            raise SystemExit(f"missing {source}")
        rows = load(source)
        everything.extend(rows)
        written = write_jsonl(args.output / f"{name}.jsonl", rows)
        manifest["fixtures"][name] = {
            "rows": len(rows),
            "groups": len({row.group for row in rows}),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "converted_sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
            "max_state_bytes": max(len(row.state.encode()) for row in rows),
            "max_question_bytes": max(len(row.question.encode()) for row in rows),
            "max_option_bytes": max(len(option.encode()) for row in rows
                                    for option in row.options),
        }

    if args.split_groups:
        splits = split_by_group(everything, (0.6, 0.15, 0.25), args.seed)
        manifest["grouped_split"] = {}
        for name, rows in splits.items():
            written = write_jsonl(args.output / f"split-{name}.jsonl", rows)
            manifest["grouped_split"][name] = {
                "rows": len(rows),
                "groups": len({row.group for row in rows}),
                "sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
            }
        overlap = (
            {row.group for row in splits["train"]}
            & {row.group for row in splits["test"]}
        )
        assert not overlap, f"group leaked across the split: {overlap}"

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
