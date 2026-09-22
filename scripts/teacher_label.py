#!/usr/bin/env python3
"""Label decision rows with a teacher LLM's probability over the options.

Distillation target for `blink_train.train --distill-alpha`. For every row the
teacher reads the state, the question and the options lettered A, B, C, ... and
we read its next-token distribution over those letters after "Answer:". No text
is generated, so one row costs one forward pass per ordering.

The prompt can open with a task preamble (`--preamble`, e.g.
scripts/teacher_prompts/nli.txt) and a few solved examples drawn from the
training split (`--shots K`: K rows per gold option text, balanced, short
states only, fixed order). On WANLI these took MiniCPM5-2B from 0.44 to 0.57
top-1 on 600 validation rows; without them the teacher is weaker than the
student it is meant to teach.

A small model prefers some letters to others whatever the content. Each row is
therefore scored once per cyclic rotation of its options and the
probabilities, mapped back to the original order, are averaged; a letter bias
then falls equally on every option instead of on whichever option happens to
sit at "A". WANLI lists its three options in the same order on every row, so
without this the bias would be a systematic error, not noise.

It also prefers some *answers*: on WANLI it says "supports" far more often
than the labels do. `--prior-correction` removes that without looking at a
label: within each split, every option text that recurs often enough gets its
teacher probability divided by the teacher's mean probability for that text,
and the row is renormalised (the "calibrate before use" correction). The
uncorrected distribution is kept as `target_raw`.

The output is the input rows, unchanged, with added fields:

    "target":     [p_0, p_1, ...]   teacher probability per option, summing to 1
    "target_raw": [...]             the same before --prior-correction

and a manifest next to it with the model, its pinned revision, the prompt, the
checksums and the teacher's own accuracy and calibration on every labelled
split. The teacher never sees a label; the gold label is only read afterwards
to report how good the teacher is.

Runs in its own environment, because MLX is not a training dependency:

    uv venv --python 3.12 .venv-teacher
    uv pip install --python .venv-teacher/bin/python mlx-lm huggingface_hub
    .venv-teacher/bin/python scripts/teacher_label.py \\
        artifacts/external/teacher/MiniCPM5-2B data/wanli \\
        --splits train validation test --output data/wanli-minicpm5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"

INSTRUCTION = "Reply with the letter of the correct option only."


def render(row: dict, order: list[int]) -> str:
    options = "\n".join(f"{LETTERS[slot]}. {row['options'][index]}"
                        for slot, index in enumerate(order))
    return (f"State:\n{row['state']}\n\nQuestion:\n{row['question']}\n\n"
            f"Options:\n{options}")


class Prompt:
    """Preamble, solved examples, then the row. The examples are shown in their
    stored option order with the gold letter; they come from the training split
    only, and they are the only place a label reaches the teacher."""

    def __init__(self, preamble: str = "", shots: list[dict] | None = None) -> None:
        self.preamble = preamble.strip()
        self.shots = shots or []

    def body(self, row: dict, order: list[int]) -> str:
        parts = []
        if self.preamble:
            parts.append(self.preamble)
        if self.shots:
            solved = [render(shot, list(range(len(shot["options"]))))
                      + f"\nAnswer: {LETTERS[shot['label']]}" for shot in self.shots]
            parts.append("Examples:\n\n" + "\n\n".join(solved))
            parts.append("Now this one.")
        parts.append(render(row, order) + "\n\n" + INSTRUCTION)
        return "\n\n".join(parts)


def pick_shots(rows: list[dict], per_answer: int, max_state: int = 220,
               skip: int = 1000, seed: int = 3) -> list[dict]:
    """`per_answer` rows per gold option text, from row `skip` on, short
    states only, then shuffled with a fixed seed so no answer sits last."""
    import random

    taken, seen = [], {}
    for row in rows[skip:]:
        answer = row["options"][row["label"]]
        if seen.get(answer, 0) < per_answer and len(row["state"]) < max_state:
            taken.append(row)
            seen[answer] = seen.get(answer, 0) + 1
    random.Random(seed).shuffle(taken)
    return taken


def prior_correct(rows: list[dict], targets: list[list[float]],
                  min_count: int = 50) -> list[list[float]]:
    """Divide each probability by the teacher's mean probability for that
    option text over the split, for texts seen at least `min_count` times,
    and renormalise. Uses no label."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row, target in zip(rows, targets):
        for option, p in zip(row["options"], target):
            totals[option] = totals.get(option, 0.0) + p
            counts[option] = counts.get(option, 0) + 1
    corrected = []
    for row, target in zip(rows, targets):
        scaled = [p / (totals[o] / counts[o]) if counts[o] >= min_count else p
                  for o, p in zip(row["options"], target)]
        total = sum(scaled)
        corrected.append([value / total for value in scaled])
    return corrected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Teacher:
    def __init__(self, path: str, batch_tokens: int, prompt: Prompt) -> None:
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.model, self.tokenizer = load(path)
        self.batch_tokens = batch_tokens
        self.prompt = prompt
        # The letter as it follows "Answer:", i.e. with its leading space.
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.tokenizer.encode(" " + letter, add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit(f"' {letter}' is not a single token")
            self.letter_ids.append(ids[0])
        pad = self.tokenizer.pad_token_id
        self.pad = pad if pad is not None else 0

        self.prefix: list[int] | None = None
        self.prefix_cache = None

    def encode(self, row: dict, order: list[int]) -> list[int]:
        """Token ids of the whole prompt, minus the shared prefix.

        Everything before the row's own "State:" -- the chat header, the
        preamble and the solved examples -- is the same for every row, so it
        is run through the model once and its key/value cache reused. The
        split is checked on every row: the full prompt must tokenize to the
        prefix followed by the rest, or the reuse would not be exact."""
        messages = [{"role": "user", "content": self.prompt.body(row, order)}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False) + "Answer:"
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.prefix is None:
            cut = text.rindex("State:\n")
            self.prefix = self.tokenizer.encode(text[:cut], add_special_tokens=False)
        if ids[:len(self.prefix)] != self.prefix:
            raise SystemExit("the shared prompt prefix does not tokenize "
                             "the same inside a full prompt")
        return ids[len(self.prefix):]

    def caches(self, batch: int):
        """A key/value cache per layer holding the shared prefix, repeated
        for `batch` rows. The prefix is computed once, on first use."""
        mx = self.mx
        from mlx_lm.models.cache import KVCache, make_prompt_cache

        if self.prefix_cache is None:
            cache = make_prompt_cache(self.model)
            self.model(mx.array([self.prefix]), cache=cache)
            self.prefix_cache = [layer.state for layer in cache]
            mx.eval(self.prefix_cache)
        out = []
        for keys, values in self.prefix_cache:
            layer = KVCache()
            layer.state = (mx.repeat(keys, batch, axis=0),
                           mx.repeat(values, batch, axis=0))
            out.append(layer)
        return out

    def letter_logprobs(self, sequences: list[list[int]], count: int) -> list[list[float]]:
        """Log-probabilities of the first `count` letters, renormalised over
        them, at the last position of each sequence.

        Each sequence continues the cached shared prefix. Sequences are
        right-padded; attention is causal, so padding after a sequence's last
        token cannot change what that token sees, and reading the logits at
        each sequence's own last position is exact."""
        mx = self.mx
        width = max(len(s) for s in sequences)
        batch = mx.array([s + [self.pad] * (width - len(s)) for s in sequences])
        logits = self.model(batch, cache=self.caches(len(sequences))).astype(mx.float32)
        last = mx.array([len(s) - 1 for s in sequences])
        picked = logits[mx.arange(len(sequences)), last]
        letters = picked[:, mx.array(self.letter_ids[:count])]
        letters = letters - mx.logsumexp(letters, axis=-1, keepdims=True)
        mx.eval(letters)
        return letters.tolist()

    def label(self, rows: list[dict], progress: str) -> list[list[float]]:
        """Teacher distribution per row, averaged over the cyclic rotations."""
        jobs = []  # (row index, order, token ids)
        for index, row in enumerate(rows):
            n = len(row["options"])
            if n > len(LETTERS):
                raise SystemExit(f"row {index} has more than {len(LETTERS)} options")
            for shift in range(n):
                order = [(slot + shift) % n for slot in range(n)]
                jobs.append((index, order, self.encode(row, order)))
        # Similar lengths together, so a batch pads little.
        jobs.sort(key=lambda job: (len(job[2]), len(job[1])))

        sums = [[0.0] * len(row["options"]) for row in rows]
        started, done = time.perf_counter(), 0
        position = 0
        while position < len(jobs):
            n = len(jobs[position][1])
            group = []
            while (position < len(jobs) and len(jobs[position][1]) == n
                   and (len(group) + 1) * len(jobs[position][2]) <= self.batch_tokens):
                group.append(jobs[position])
                position += 1
            if not group:  # one sequence longer than the budget
                group.append(jobs[position])
                position += 1
            logprobs = self.letter_logprobs([job[2] for job in group], n)
            for (index, order, _), values in zip(group, logprobs):
                for slot, original in enumerate(order):
                    sums[index][original] += math.exp(values[slot])
            done += len(group)
            if done // 2000 != (done - len(group)) // 2000:
                rate = done / (time.perf_counter() - started)
                print(json.dumps({"split": progress, "scored": done,
                                  "of": len(jobs), "per_second": round(rate, 1)}),
                      flush=True)
        return [[value / len(row["options"]) for value in total]
                for total, row in zip(sums, rows)]


def teacher_metrics(rows: list[dict], targets: list[list[float]]) -> dict:
    """How good the teacher is against the gold labels, with the same ECE
    binning eval/evaluate.py uses (15 equal-width confidence bins)."""
    correct, nll, brier = 0, 0.0, 0.0
    bins = [[0, 0.0, 0.0] for _ in range(15)]
    for row, target in zip(rows, targets):
        label = row["label"]
        top = max(range(len(target)), key=target.__getitem__)
        hit = top == label
        correct += hit
        nll -= math.log(max(target[label], 1e-12))
        brier += sum((p - (i == label)) ** 2 for i, p in enumerate(target))
        confidence = target[top]
        b = min(14, int(confidence * 15))
        bins[b][0] += 1
        bins[b][1] += confidence
        bins[b][2] += hit
    n = len(rows)
    ece = sum(abs(c / max(k, 1) - h / max(k, 1)) * k / n for k, c, h in bins)
    return {"rows": n, "top1": correct / n, "nll": nll / n,
            "brier": brier / n, "ece": ece}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="local directory of the teacher model")
    parser.add_argument("corpus", help="directory holding <split>.jsonl")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--output", required=True, help="directory for the labelled splits")
    parser.add_argument("--limit", type=int, default=0, help="first N rows only (smoke)")
    parser.add_argument("--batch-tokens", type=int, default=12288)
    parser.add_argument("--revision", default="", help="recorded in the manifest")
    parser.add_argument("--preamble", default=None, help="task description file")
    parser.add_argument("--shots", type=int, default=0,
                        help="solved training examples per gold option text")
    parser.add_argument("--prior-correction", action="store_true")
    args = parser.parse_args()

    preamble = Path(args.preamble).read_text() if args.preamble else ""
    shots = []
    if args.shots:
        train = [json.loads(line) for line in
                 (Path(args.corpus) / "train.jsonl").read_text().splitlines() if line.strip()]
        shots = pick_shots(train, args.shots)
    prompt = Prompt(preamble, shots)
    teacher = Teacher(args.model, args.batch_tokens, prompt)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator": "scripts/teacher_label.py",
        "teacher": {"model": Path(args.model).name, "revision": args.revision,
                    "weights_sha256": {p.name: sha256(p) for p in
                                       sorted(Path(args.model).glob("*.safetensors"))}},
        "prompt": {"instruction": INSTRUCTION, "suffix": "Answer:",
                   "thinking": False, "orderings": "every cyclic rotation, averaged",
                   "preamble": preamble.strip(), "shots_per_answer": args.shots,
                   "prior_correction": args.prior_correction,
                   "example": prompt.body({"state": "<state>", "question": "<question>",
                                           "options": ["<option 0>", "<option 1>"]},
                                          [0, 1])},
        "splits": {},
    }
    for split in args.splits:
        source = Path(args.corpus) / f"{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if args.limit:
            rows = rows[:args.limit]
        started = time.perf_counter()
        raw = teacher.label(rows, split)
        seconds = time.perf_counter() - started
        targets = prior_correct(rows, raw) if args.prior_correction else raw
        destination = output / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for row, target, before in zip(rows, targets, raw):
                handle.write(json.dumps({**row, "target": target, "target_raw": before},
                                        ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "source": str(source), "source_sha256": sha256(source),
            "output_sha256": sha256(destination), "seconds": round(seconds, 1),
            "teacher_vs_gold": teacher_metrics(rows, targets),
            "teacher_raw_vs_gold": teacher_metrics(rows, raw),
        }
        print(json.dumps({split: manifest["splits"][split]}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
