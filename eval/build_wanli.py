#!/usr/bin/env python3
"""Build Blink decision rows from WANLI.

WANLI (Liu et al., 2022, CC-BY-4.0) is a worker-and-AI-collaborative natural
language inference set. It is the external, non-synthetic check in this
repository: the premise becomes the state, the hypothesis becomes the question,
and the three NLI relations become three described options. SemIf uses 256
WANLI rows as its external check, so the task framing is comparable even though
the row selection is not.

Unlike the SemIf fixtures, WANLI is large enough to train on, which is what
makes it a real generalisation test for a model this small rather than a
transfer test it is bound to fail.

Splitting: WANLI ships a train and a test split. Rows are grouped by premise so
that two hypotheses about one premise cannot straddle the split, and the
validation split is carved out of train.

    python eval/build_wanli.py --output data/wanli
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from blink_train.data import Row, write_jsonl  # noqa: E402

# The option text the model actually reads. Order is fixed here and shuffled
# per row so the label index carries no signal.
RELATIONS = {
    "entailment": "the premise supports the hypothesis",
    "neutral": "the premise neither supports nor contradicts the hypothesis",
    "contradiction": "the premise contradicts the hypothesis",
}
ORDER = ("entailment", "neutral", "contradiction")

REVISION = "61c95318fd71c55b6ba355d76253254615f387ec"
URL = ("https://huggingface.co/datasets/alisawuffles/WANLI/resolve/"
       f"{REVISION}/{{split}}.jsonl")


def download(split: str, cache: Path) -> Path:
    import urllib.request

    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"wanli-{split}.jsonl"
    if target.exists():
        return target
    url = URL.format(split=split)
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())
    return target


def to_row(payload: dict, rng: random.Random) -> Row | None:
    premise = (payload.get("premise") or "").strip()
    hypothesis = (payload.get("hypothesis") or "").strip()
    gold = (payload.get("gold") or payload.get("label") or "").strip().lower()
    if not premise or not hypothesis or gold not in RELATIONS:
        return None
    if len(premise) > 1000 or len(hypothesis) > 500:
        return None

    order = list(ORDER)
    rng.shuffle(order)
    return Row(
        state=premise,
        question=f"Assess the hypothesis: {hypothesis}",
        options=tuple(RELATIONS[name] for name in order),
        label=order.index(gold),
        group=hashlib.sha256(premise.encode()).hexdigest()[:16],
        family="nli",
    )


def read(path: Path, rng: random.Random) -> list[Row]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = to_row(json.loads(line), rng)
        if row is not None:
            rows.append(row)
    return rows


def carve_validation(rows: list[Row], fraction: float, seed: int):
    by_group: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_group[row.group].append(row)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    cut = int(len(groups) * fraction)
    validation = [row for group in groups[:cut] for row in by_group[group]]
    train = [row for group in groups[cut:] for row in by_group[group]]
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/wanli"))
    parser.add_argument("--cache", type=Path, default=Path("data/.cache"))
    parser.add_argument("--max-train", type=int, default=40000)
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_rows = read(download("train", args.cache), rng)
    test_rows = read(download("test", args.cache), rng)

    train_groups = {row.group for row in train_rows}
    test_rows = [row for row in test_rows if row.group not in train_groups]

    train_rows = train_rows[:args.max_train]
    test_rows = test_rows[:args.max_test]
    train_rows, validation_rows = carve_validation(
        train_rows, args.validation_fraction, args.seed)

    splits = {"train": train_rows, "validation": validation_rows, "test": test_rows}
    manifest = {
        "source": "WANLI",
        "upstream": "https://huggingface.co/datasets/alisawuffles/WANLI",
        "revision": REVISION,
        "licence": "CC-BY-4.0; see the dataset card",
        "citation": "Liu et al., WANLI: Worker and AI Collaboration for "
                    "Natural Language Inference Dataset Creation, 2022",
        "mapping": "premise -> state, hypothesis -> question, "
                   "entailment/neutral/contradiction -> three described options, "
                   "option order shuffled per row",
        "grouping": "rows are grouped by premise; groups never straddle a split",
        "splits": {},
    }
    for name, rows in splits.items():
        written = write_jsonl(args.output / f"{name}.jsonl", rows)
        manifest["splits"][name] = {
            "rows": len(rows),
            "groups": len({row.group for row in rows}),
            "sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
            "label_counts": dict(Counter(row.label for row in rows)),
            "max_state_bytes": max((len(r.state.encode()) for r in rows), default=0),
            "max_question_bytes": max((len(r.question.encode()) for r in rows), default=0),
        }

    overlap = ({row.group for row in splits["train"]}
               & {row.group for row in splits["test"]})
    assert not overlap, "premise leaked between train and test"

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
