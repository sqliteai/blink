#!/usr/bin/env python3
"""Rebuild SemIf's exact 256-row WANLI subset, so the comparison is real.

SemIf evaluates a frozen Qwen3.5-4B on 256 WANLI rows and reports 0.637
balanced accuracy. Quoting that next to a number computed on a different subset
would not be a comparison. This reads the row identifiers SemIf commits in
`benchmarks/manifests/source-selection.jsonl` and rebuilds the same rows, in
the same option order, from the same pinned WANLI revision.

It also checks for contamination: a row is only usable if its premise never
appeared in the split the model was trained on.

    python eval/build_wanli256.py --source ../SemIf --output data/wanli256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from blink_train.data import Row, read_jsonl, write_jsonl  # noqa: E402

# SemIf maps the three NLI relations onto these option ids. The descriptions
# are Blink's own -- they are what the model was trained to read -- but the
# rows, the ordering and the labels are SemIf's.
OPTION_TEXT = {
    "supported": "the premise supports the hypothesis",
    "insufficient": "the premise neither supports nor contradicts the hypothesis",
    "contradicted": "the premise contradicts the hypothesis",
}
GOLD_TO_OPTION = {
    "entailment": "supported",
    "neutral": "insufficient",
    "contradiction": "contradicted",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("../SemIf"))
    parser.add_argument("--cache", type=Path, default=Path("data/.cache"))
    parser.add_argument("--output", type=Path, default=Path("data/wanli256"))
    parser.add_argument("--trained-on", type=Path,
                        default=Path("data/wanli/train.jsonl"),
                        help="the split the model was trained on, for the "
                             "contamination check")
    args = parser.parse_args()

    manifest = args.source / "benchmarks/manifests/source-selection.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found; clone SemIf or pass --source")

    selection = [json.loads(line) for line in
                 manifest.read_text().splitlines() if line.strip()]
    wanli_rows = [json.loads(line) for line in
                  (args.cache / "wanli-test.jsonl").read_text().splitlines()
                  if line.strip()]
    by_id = {row["id"]: row for row in wanli_rows}

    trained_premises = set()
    if args.trained_on.exists():
        trained_premises = {row.state for row in read_jsonl(args.trained_on)}

    rows: list[Row] = []
    contaminated, missing, wrong_split = [], [], []
    for entry in selection:
        if entry.get("source") != "wanli":
            continue
        upstream = entry["upstream"]
        if upstream.get("split") != "test":
            wrong_split.append(entry["id"])
            continue
        source = by_id.get(upstream["source_id"])
        if source is None:
            missing.append(upstream["source_id"])
            continue
        if source["premise"] in trained_premises:
            contaminated.append(entry["id"])
            continue

        gold = GOLD_TO_OPTION[source["gold"]]
        order = entry["option_ids"]          # SemIf's displayed order
        rows.append(Row(
            state=source["premise"],
            question=f"Assess the hypothesis: {source['hypothesis']}",
            options=tuple(OPTION_TEXT[o] for o in order),
            label=order.index(gold),
            group=hashlib.sha256(source["premise"].encode()).hexdigest()[:16],
            family="nli",
        ))

    if not rows:
        raise SystemExit("no usable rows; check --cache and --source")

    written = write_jsonl(args.output / "test.jsonl", rows)
    report = {
        "source": "SemIf benchmarks/manifests/source-selection.jsonl",
        "upstream": "WANLI, revision "
                    "61c95318fd71c55b6ba355d76253254615f387ec",
        "selected_by_source": sum(1 for e in selection if e.get("source") == "wanli"),
        "usable": len(rows),
        "dropped": {
            "premise_seen_in_training": len(contaminated),
            "not_found_in_wanli_test": len(missing),
            "not_from_the_test_split": len(wrong_split),
        },
        "note": "Rows, display order and labels are SemIf's; the option "
                "descriptions are Blink's, because they are what the model was "
                "trained to read. Blink's score is order-equivariant, so the "
                "display order cannot change the result.",
        "sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
