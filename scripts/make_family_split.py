#!/usr/bin/env python3
"""Write single-family subsets of the synthetic corpus.

The mixed corpus averages four unlike tasks, which makes it hard to see whether
a family is unlearnable or merely slow to learn. Training on one family answers
that directly, and it is the experiment behind the FiLM claim in
docs/ARCHITECTURE.md.

    python scripts/make_family_split.py --family selection --output data/selection
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from blink_train.data import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--source", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {"family": args.family, "source": str(args.source), "splits": {}}
    for split in ("train", "validation", "test"):
        rows = [row for row in read_jsonl(args.source / f"{split}.jsonl")
                if row.family == args.family]
        if not rows:
            raise SystemExit(f"no {args.family} rows in {split}")
        path = write_jsonl(args.output / f"{split}.jsonl", rows)
        manifest["splits"][split] = {
            "rows": len(rows),
            "groups": len({row.group for row in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
