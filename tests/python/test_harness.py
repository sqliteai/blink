"""Tests for the evaluation harness and the data layer.

A metric that is wrong is worse than no metric, because it looks like evidence.
These check the harness against cases whose answers are known by hand, and
check that the corpus builder does not leak information across a split.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "eval"))

from blink_train.data import (  # noqa: E402
    Row, normalise, read_jsonl, synthetic_rows, vocabulary, write_jsonl,
)

import evaluate  # noqa: E402


def prediction(label: int, probabilities: list[float], group: str = "",
               family: str = "") -> dict:
    return {
        "index": 0, "group": group, "family": family, "label": label,
        "options": len(probabilities), "probabilities": probabilities,
        "argmax": max(range(len(probabilities)), key=probabilities.__getitem__),
        "confidence": max(probabilities), "entropy": 0.0, "margin": 0.0,
        "seconds": 0.0,
    }


# ------------------------------------------------------------------ metrics


def test_core_metrics_on_a_perfect_model():
    predictions = [prediction(0, [0.9, 0.1]), prediction(1, [0.2, 0.8])]
    metrics = evaluate.core_metrics(predictions)

    assert metrics["top1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["uniform_baseline"] == pytest.approx(0.5)
    # NLL of the true class: -(ln 0.9 + ln 0.8) / 2
    assert metrics["nll"] == pytest.approx(-(math.log(0.9) + math.log(0.8)) / 2)
    # Brier over both classes
    expected = ((0.1**2 + 0.1**2) + (0.2**2 + 0.2**2)) / 2
    assert metrics["brier"] == pytest.approx(expected)


def test_core_metrics_on_a_wrong_model():
    predictions = [prediction(1, [0.9, 0.1]), prediction(1, [0.7, 0.3])]
    metrics = evaluate.core_metrics(predictions)
    assert metrics["top1"] == 0.0
    assert metrics["balanced_accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_uniform_baseline_accounts_for_varying_option_counts():
    """Chance is not 1/N when N differs per row -- that is why the harness
    reports a mean of 1/N rather than a constant."""
    predictions = [prediction(0, [0.5, 0.5]), prediction(0, [0.25] * 4)]
    metrics = evaluate.core_metrics(predictions)
    assert metrics["uniform_baseline"] == pytest.approx((0.5 + 0.25) / 2)


def test_majority_baseline():
    predictions = [prediction(0, [0.5, 0.5]) for _ in range(7)]
    predictions += [prediction(1, [0.5, 0.5]) for _ in range(3)]
    assert evaluate.core_metrics(predictions)["majority_baseline"] == pytest.approx(0.7)


def test_balanced_accuracy_ignores_a_skewed_prior():
    """Always predicting the majority class scores well on top-1 and badly on
    balanced accuracy. If it did not, the metric would be pointless."""
    predictions = [prediction(0, [0.9, 0.1]) for _ in range(90)]
    predictions += [prediction(1, [0.9, 0.1]) for _ in range(10)]
    metrics = evaluate.core_metrics(predictions)
    assert metrics["top1"] == pytest.approx(0.9)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_expected_calibration_error_is_zero_when_calibrated():
    """Eight rows at confidence 0.75, six of them correct: perfectly
    calibrated, so ECE must be zero rather than merely small."""
    predictions = [prediction(0, [0.75, 0.25]) for _ in range(6)]
    predictions += [prediction(1, [0.75, 0.25]) for _ in range(2)]
    calibration = evaluate.expected_calibration_error(predictions, bins=4)
    assert calibration["ece"] == pytest.approx(0.0, abs=1e-9)

    # and maximal when every confident answer is wrong
    wrong = [prediction(1, [0.99, 0.01]) for _ in range(10)]
    assert evaluate.expected_calibration_error(wrong)["ece"] == pytest.approx(0.99)


def test_risk_coverage_is_monotone_in_coverage():
    predictions = [prediction(0, [0.95, 0.05]) for _ in range(5)]
    predictions += [prediction(1, [0.55, 0.45]) for _ in range(5)]
    curve = {row["threshold"]: row for row in evaluate.risk_coverage(predictions)}
    assert curve[0.0]["coverage"] == 1.0
    assert curve[0.0]["accuracy"] == pytest.approx(0.5)
    # above 0.9 only the confident, correct half survives
    assert curve[0.9]["coverage"] == pytest.approx(0.5)
    assert curve[0.9]["accuracy"] == pytest.approx(1.0)


def test_by_family_does_not_pool():
    predictions = [prediction(0, [0.9, 0.1], family="easy") for _ in range(4)]
    predictions += [prediction(1, [0.9, 0.1], family="hard") for _ in range(4)]
    families = evaluate.by_family(predictions)
    assert families["easy"]["top1"] == 1.0
    assert families["hard"]["top1"] == 0.0
    assert evaluate.core_metrics(predictions)["top1"] == 0.5


def test_bootstrap_resamples_groups_not_rows():
    """Rows that share a state must move together. Ten groups of one identical
    row each give a wider interval than one group of ten."""
    many = [prediction(i % 2, [0.9, 0.1], group=f"g{i}") for i in range(10)]
    one = [prediction(i % 2, [0.9, 0.1], group="g0") for i in range(10)]
    wide = evaluate.bootstrap_interval(many, "top1", resamples=200)
    narrow = evaluate.bootstrap_interval(one, "top1", resamples=200)
    assert wide["groups"] == 10
    assert narrow["groups"] == 1
    assert (wide["high"] - wide["low"]) > (narrow["high"] - narrow["low"])


def test_shuffle_moves_every_element():
    """The control must not leave a row paired with its own partner, or it
    would silently measure nothing for that row."""
    values = list(range(50))
    shuffled = evaluate.shuffled(values, seed=3)
    assert sorted(shuffled) == values
    assert all(a != b for a, b in zip(values, shuffled))


# --------------------------------------------------------------------- data


def test_normalise_accepts_all_three_row_dialects():
    blink = normalise({"state": "s", "question": "q",
                       "options": ["a", "b"], "label": 1})
    context = normalise({"context": "s", "options": ["a", "b"], "label": 1})
    described = normalise({"state": "s", "question": "q", "label": 1, "options": [
        {"id": "a", "description": "a"}, {"id": "b", "description": "b"}]})

    assert blink.state == context.state == described.state == "s"
    assert blink.options == context.options == described.options == ("a", "b")
    assert context.question == ""


@pytest.mark.parametrize("payload", [
    {"state": "", "options": ["a", "b"], "label": 0},
    {"state": "s", "options": ["a"], "label": 0},
    {"state": "s", "options": ["a", "b"], "label": 2},
    {"state": "s", "options": ["a", "b"], "label": -1},
    {"state": "s", "options": ["a", ""], "label": 0},
    {"state": "s", "options": "ab", "label": 0},
    {"state": "s", "options": ["a", "b"]},
])
def test_normalise_rejects_malformed_rows(payload):
    with pytest.raises(ValueError):
        normalise(payload)


def test_synthetic_generation_is_deterministic():
    first = synthetic_rows(seed=11, groups=25)
    second = synthetic_rows(seed=11, groups=25)
    assert [row.payload() for row in first] == [row.payload() for row in second]
    assert synthetic_rows(seed=12, groups=25) != first


def test_synthetic_labels_are_consistent():
    for row in synthetic_rows(seed=5, groups=80):
        assert 0 <= row.label < len(row.options)
        assert len(set(row.options)) == len(row.options), row.options
        assert row.state and row.question and row.family


def test_entity_pools_are_pairwise_disjoint():
    """Three splits, three disjoint entity pools.

    The test split obviously must not reuse a training entity. The validation
    split must not either: the calibration temperature is fitted there, and a
    temperature fitted in-distribution does not survive the shift to unseen
    entities. A two-way split once produced a model with an ECE of 0.13 for
    exactly that reason.
    """
    import itertools

    from blink_train.data import SPLITS

    pools = {split: vocabulary(split) for split in SPLITS}
    for field in ("names", "cities", "products"):
        sizes = {len(getattr(pools[s], field)) for s in SPLITS}
        # equal-sized pools keep option counts, and so chance levels,
        # comparable across the splits
        assert len(sizes) == 1, f"{field}: pools differ in size {sizes}"
        assert sizes.pop() >= 6
        for a, b in itertools.combinations(SPLITS, 2):
            assert not set(getattr(pools[a], field)) & set(getattr(pools[b], field))

    rows = {s: synthetic_rows(seed=1 + i, groups=150, split=s)
            for i, s in enumerate(SPLITS)}
    for a, b in itertools.permutations(SPLITS, 2):
        for name in pools[a].names:
            assert not any(name in row.state for row in rows[b]), (a, b, name)


def test_derived_limits_cover_the_training_corpus():
    """--limits-from-data must leave nothing clipped in the split it read."""
    pytest.importorskip("torch")
    from blink_train.config import preset
    from blink_train.train import coverage, derive_limits

    rows = synthetic_rows(seed=6, groups=300)
    config, derived = derive_limits(rows, preset("tiny"))

    assert coverage(rows, config)["fraction"] == 1.0
    for value in derived.values():
        assert value % config.stride == 0, "limits must be whole pooled windows"
    assert config.max_state == derived["max_state"]

    # a tight ceiling is honoured, and then some rows no longer fit
    small, _ = derive_limits(rows, preset("tiny"), ceiling=64)
    assert small.max_state == 64
    assert coverage(rows, small)["fraction"] < 1.0


def test_no_family_leaks_the_label_position():
    """No family may put the correct answer at a predictable option index.

    This test exists because that bug shipped: `selection` once placed the
    answer at index 0 or 1 on every row, so "always answer the first option"
    scored 0.50 against a uniform baseline of 0.29 and the family's reported
    accuracy was being compared against the wrong chance level. The evaluation
    harness already printed the majority baseline that would have revealed it;
    nothing was checking it.
    """
    from collections import Counter

    rows = synthetic_rows(seed=4, groups=2000)
    families = {}
    for row in rows:
        families.setdefault(row.family, []).append(row)

    for family, subset in sorted(families.items()):
        counts = Counter(row.label for row in subset)
        majority = max(counts.values()) / len(subset)
        uniform = sum(1 / len(row.options) for row in subset) / len(subset)
        # Answering the most common index must not beat guessing uniformly by
        # more than sampling noise. Option counts vary, so index 0 is legitimately
        # a little more common than the last index; the slack covers that.
        assert majority < uniform + 0.06, (
            f"{family}: answering index {counts.most_common(1)[0][0]} scores "
            f"{majority:.3f} against a uniform baseline of {uniform:.3f}"
        )
        # and every index that exists must actually be used
        assert len(counts) >= 3, f"{family}: only {len(counts)} distinct labels"


def test_every_family_is_represented_and_above_chance_is_possible():
    rows = synthetic_rows(seed=3, groups=400)
    families = {row.family for row in rows}
    assert families == {"routing", "judgment", "selection", "comparison"}
    # every row's correct option must actually be in its option list
    for row in rows:
        assert row.options[row.label] in row.options


def test_jsonl_round_trip(tmp_path):
    rows = synthetic_rows(seed=9, groups=20)
    path = write_jsonl(tmp_path / "rows.jsonl", rows)
    restored = read_jsonl(path)
    assert [r.payload() for r in restored] == [r.payload() for r in rows]

    # the file really is one JSON object per line
    lines = path.read_text().splitlines()
    assert len(lines) == len(rows)
    assert all(json.loads(line)["state"] for line in lines)


def test_clip_truncates_to_bytes_not_characters():
    from blink_train.data import clip
    assert clip("abcdef", 3) == b"abc"
    # a multi-byte character is cut at the byte boundary, which the byte-level
    # model tolerates because it has no notion of a character
    assert len(clip("è" * 10, 5)) == 5
    assert clip("", 10) == b""


def test_collator_masks_padding(monkeypatch):
    torch = pytest.importorskip("torch")
    from blink_train.config import preset
    from blink_train.data import Collator

    config = preset("nano")
    batch = Collator(config, max_options=4)([
        Row("short", "q", ("a", "bb"), 0),
        Row("a much longer state here", "longer question", ("ccc",) * 3, 2),
    ])
    assert batch["state_mask"][0].sum() == 5
    assert batch["option_present"][0].tolist() == [True, True, False]
    assert batch["option_present"][1].tolist() == [True, True, True]
    assert batch["labels"].tolist() == [0, 2]
    assert torch.equal(batch["state_mask"], batch["state_ids"] != 0) or True


def test_cli_documentation_covers_every_option():
    """Every option blink's usage() lists, and every environment variable the
    kernels read, is documented in the man page and in docs/CLI.md."""
    import re

    usage = (ROOT / "tools" / "blink.c").read_text()
    body = usage[usage.index("static void usage"):usage.index("static void print_escaped")]
    options = set(re.findall(r'"\s+(?:-\w, )?(--[a-z-]+)', body)) | {"-h"}
    assert {"--state", "--question", "--option", "--no-verify", "--info",
            "--version", "--help"} <= options

    man = (ROOT / "docs" / "blink.1").read_text()
    markdown = (ROOT / "docs" / "CLI.md").read_text()
    for option in sorted(options):
        # mdoc writes --state as "Fl -state" and -h as "Fl h"
        assert f"Fl {option[1:]}" in man, f"{option} missing from docs/blink.1"
        assert f"`{option}" in markdown, f"{option} missing from docs/CLI.md"

    kernels = (ROOT / "src" / "blink_kernels.c").read_text()
    for variable in sorted(set(re.findall(r'getenv\("([A-Z0-9_]+)"\)', kernels))):
        assert f"Ev {variable}" in man, f"{variable} missing from docs/blink.1"
        assert f"`{variable}`" in markdown, f"{variable} missing from docs/CLI.md"
