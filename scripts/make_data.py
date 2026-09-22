#!/usr/bin/env python3
"""Build the deterministic synthetic corpus.

Splits are made on scenario groups, never on rows: several questions can share
one state, and letting them straddle a split would leak the state into the test
set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from blink_train.data import write_synthetic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--train-groups", type=int, default=20000)
    parser.add_argument("--validation-groups", type=int, default=2000)
    parser.add_argument("--test-groups", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    counts = write_synthetic(args.output, {
        "train": args.train_groups,
        "validation": args.validation_groups,
        "test": args.test_groups,
    }, args.seed)

    digests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.glob("*.jsonl"))
    }
    manifest = {
        "generator": "blink_train.data.write_synthetic",
        "seed": args.seed,
        "groups": {"train": args.train_groups,
                   "validation": args.validation_groups,
                   "test": args.test_groups},
        "rows": counts,
        "sha256": digests,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
