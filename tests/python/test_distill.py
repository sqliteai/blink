"""Teacher targets: parsing, batching and the distillation loss.

The loss must reduce to the plain one exactly, not approximately: at
--distill-alpha 0 it is the plain cross-entropy by construction, and a one-hot
target at alpha 1 must give the same number, so a distilled run differs from a
gold run only by what the targets say. SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

torch = pytest.importorskip("torch")

from blink_train.config import preset  # noqa: E402
from blink_train.data import Collator, check_target, normalise  # noqa: E402
from blink_train.train import soft_cross_entropy, training_loss  # noqa: E402
from torch.nn import functional as F  # noqa: E402

ROW = {"state": "s", "question": "q", "options": ["a", "b", "c"], "label": 1}


def test_target_is_optional_and_round_trips():
    assert normalise(ROW).target is None
    row = normalise({**ROW, "target": [0.2, 0.5, 0.3]})
    assert row.target == pytest.approx((0.2, 0.5, 0.3))
    assert normalise(row.payload()).target == row.target
    assert "target" not in normalise(ROW).payload()


def test_target_rounding_error_is_divided_out():
    target = check_target([0.33333, 0.33333, 0.33333], 3)
    assert math.fsum(target) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("target", [
    [0.5, 0.5],                 # wrong length
    [0.5, 0.6, -0.1],           # negative
    [0.2, 0.2, 0.2],            # does not sum to 1
    [0.5, float("nan"), 0.5],   # not finite
    [True, 0.0, 0.0],           # not a number
    "0.2 0.5 0.3",              # not a list
])
def test_malformed_targets_are_refused(target):
    with pytest.raises(ValueError):
        normalise({**ROW, "target": target})


def test_collator_carries_targets_only_when_every_row_has_one():
    collate = Collator(preset("nano"))
    with_target = normalise({**ROW, "target": [0.1, 0.7, 0.2]})
    two = normalise({"state": "s", "question": "q", "options": ["x", "y"],
                     "label": 0, "target": [0.9, 0.1]})
    batch = collate([with_target, two])
    assert torch.allclose(batch["targets"], torch.tensor([[0.1, 0.7, 0.2],
                                                          [0.9, 0.1, 0.0]]))
    assert "targets" not in collate([with_target, normalise(ROW)])


def test_alpha_zero_is_the_plain_loss_exactly():
    torch.manual_seed(0)
    logits = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 2, 1, 0])
    batch = {"labels": labels, "targets": torch.softmax(torch.randn(5, 3), -1)}
    assert torch.equal(training_loss(logits, batch, 0.0),
                       F.cross_entropy(logits, labels))


def test_one_hot_target_at_alpha_one_is_the_plain_loss():
    torch.manual_seed(1)
    logits = torch.randn(6, 4)
    labels = torch.tensor([3, 0, 1, 2, 2, 0])
    batch = {"labels": labels, "targets": F.one_hot(labels, 4).float()}
    assert torch.allclose(training_loss(logits, batch, 1.0),
                          F.cross_entropy(logits, labels), atol=1e-6, rtol=0)


def test_absent_options_do_not_turn_the_loss_into_nan():
    minimum = torch.finfo(torch.float32).min
    logits = torch.tensor([[1.0, 0.0, minimum], [0.5, -0.5, 0.2]])
    targets = torch.tensor([[0.6, 0.4, 0.0], [0.2, 0.3, 0.5]])
    loss = soft_cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    logits.requires_grad_(True)
    soft_cross_entropy(logits, targets).backward()
    assert torch.isfinite(logits.grad).all()


def test_mixed_loss_is_the_weighted_sum():
    torch.manual_seed(2)
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 2, 1, 1])
    targets = torch.softmax(torch.randn(4, 3), -1)
    batch = {"labels": labels, "targets": targets}
    expected = 0.7 * F.cross_entropy(logits, labels) + 0.3 * soft_cross_entropy(logits, targets)
    assert torch.allclose(training_loss(logits, batch, 0.3), expected)
