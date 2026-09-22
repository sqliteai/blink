#!/usr/bin/env python3
"""Build Wikispeedia next-click rows for Blink and for jevlike, from one
selection, so the two systems are compared on identical data.

Wikispeedia is jevlike's ground: they report 26% with a frozen Qwen2.5-0.5B
encoder and 29% with a small model trained from scratch, against about 8% for
their shuffled control. Running Blink here is the comparison in the direction
that does not favour it.

The selection -- which path, which step, which links go in the menu, which
split a row lands in -- is jevlike's, reimplemented here from
`jevlike/data.py::build_wikispeedia` so that both systems see the same rows in
the same splits. Two deliberate differences, both stated because they change
what is being measured:

* **The body budget is configurable and applied to both.** jevlike's builder
  keeps 2,048 bytes of article text and its trainer then truncates the whole
  context to 192 by default, so its published runs saw the titles and almost
  none of the body. Training Blink on 2,048-byte states is affordable but slow,
  and comparing a model that reads the body against one that does not would
  measure the budget. Both formats are built at the same `--body-bytes`.
* **Blink splits the row where its architecture splits.** The navigation goal
  becomes the question and the article body becomes the state; jevlike gets the
  same bytes in the single context it accepts. This is the same courtesy as
  sizing jevlike's `--context-tokens` from the data: each system receives the
  same information in the shape it was built for.

    python eval/build_wikispeedia.py --root data/wikispeedia
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from blink_train.data import Row, write_jsonl  # noqa: E402


def stable(text: str) -> int:
    """jevlike's hash, so the selection and the splits match theirs."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def title(text: str) -> str:
    return unquote(text).replace("_", " ")


def build(root: Path, body_bytes: int, max_options: int):
    graph = root / "wikispeedia_paths-and-graph"
    outgoing: dict[str, list[str]] = {}
    for line in (graph / "links.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            source, target = line.split("\t")
            outgoing.setdefault(source, []).append(target)

    bodies: dict[str, str] = {}

    def body_of(article: str) -> str:
        if article not in bodies:
            path = root / "plaintext_articles" / f"{article}.txt"
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            bodies[article] = " ".join(text.split())
        return bodies[article]

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    skipped = Counter()

    lines = (graph / "paths_finished.tsv").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        path: list[str] = []
        for node in fields[3].split(";"):
            if node == "<":
                if len(path) > 1:
                    path.pop()
            else:
                path.append(node)
        if len(path) < 2:
            skipped["path_too_short"] += 1
            continue

        step = stable(f"{fields[0]}:{fields[1]}:{index}") % (len(path) - 1)
        current, click, target = path[step], path[step + 1], path[-1]
        candidates = list(dict.fromkeys(outgoing.get(current, ())))
        if len(candidates) < 2 or click not in candidates:
            skipped["no_usable_menu"] += 1
            continue

        rng = random.Random(stable(f"{index}:{target}:menu"))
        others = [item for item in candidates if item != click]
        rng.shuffle(others)
        menu = [click] + others[:max_options - 1]
        rng.shuffle(menu)

        body = body_of(current)[:body_bytes]
        options = tuple(title(item) for item in menu)
        label = menu.index(click)
        bucket = stable(target + ":split") % 10
        name = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        splits[name].append({
            "target": title(target), "current": title(current),
            "body": body, "options": options, "label": label,
            # Grouped by target article, matching how the split is drawn, so a
            # bootstrap resamples whole targets.
            "group": hashlib.sha256(target.encode()).hexdigest()[:16],
        })
    return splits, skipped


def to_blink(item: dict) -> Row:
    return Row(
        state=item["body"] or item["current"],
        question=f"Target article: {item['target']}\n"
                 f"Current article: {item['current']}\n"
                 f"Which link leads towards the target?",
        options=item["options"],
        label=item["label"],
        group=item["group"],
        family="next_click",
    )


def to_external(item: dict) -> dict:
    """jevlike's own context layout, with the same bytes."""
    return {
        "context": f"Target article: {item['target']}\n"
                   f"Current article: {item['current']}\n{item['body']}",
        "options": list(item["options"]),
        "label": item["label"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/wikispeedia"))
    parser.add_argument("--blink-output", type=Path,
                        default=Path("data/wikispeedia/blink"))
    parser.add_argument("--external-output", type=Path,
                        default=Path("data/wikispeedia/external"))
    parser.add_argument("--body-bytes", type=int, default=512)
    parser.add_argument("--max-options", type=int, default=32)
    args = parser.parse_args()

    splits, skipped = build(args.root, args.body_bytes, args.max_options)
    manifest = {
        "source": "Wikispeedia (SNAP)",
        "citation": "Robert West and Jure Leskovec, Human Wayfinding in "
                    "Information Networks, WWW 2012",
        "selection": "reimplements jevlike/data.py::build_wikispeedia; same "
                     "hash, same step choice, same menu sampling, same splits",
        "body_bytes": args.body_bytes,
        "max_options": args.max_options,
        "skipped": dict(skipped),
        "splits": {},
    }

    args.external_output.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        rows = [to_blink(item) for item in items]
        blink_path = write_jsonl(args.blink_output / f"{name}.jsonl", rows)
        external_path = args.external_output / f"{name}.jsonl"
        external_path.write_text(
            "\n".join(json.dumps(to_external(i), ensure_ascii=False) for i in items)
            + "\n", encoding="utf-8")
        manifest["splits"][name] = {
            "rows": len(rows),
            "groups": len({r.group for r in rows}),
            "options_min": min(len(r.options) for r in rows),
            "options_max": max(len(r.options) for r in rows),
            "options_mean": round(sum(len(r.options) for r in rows) / len(rows), 2),
            "uniform_baseline": round(
                sum(1 / len(r.options) for r in rows) / len(rows), 4),
            "max_state_bytes": max(len(r.state.encode()) for r in rows),
            "max_question_bytes": max(len(r.question.encode()) for r in rows),
            "max_option_bytes": max(len(o.encode()) for r in rows
                                    for o in r.options),
            "max_external_context_bytes": max(
                len(to_external(i)["context"].encode()) for i in items),
            "blink_sha256": hashlib.sha256(blink_path.read_bytes()).hexdigest(),
            "external_sha256": hashlib.sha256(external_path.read_bytes()).hexdigest(),
        }

    overlap = ({r["group"] for r in splits["train"]}
               & {r["group"] for r in splits["test"]})
    assert not overlap, "a target article leaked between train and test"

    (args.blink_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
