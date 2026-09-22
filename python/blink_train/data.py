"""Decision rows, batching, and the deterministic synthetic corpus.

A row is one JSON object per line:

    {"state": "...", "question": "...",
     "options": ["queue a", "queue b"], "label": 0}

`options` may also be objects with `id` and `description`, and a `context`
field is accepted as an alias for `state`. Anything the
loaders accept is normalised to the shape above before batching.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BlinkConfig


@dataclass(frozen=True)
class Row:
    state: str
    question: str
    options: tuple[str, ...]
    label: int
    group: str = ""
    family: str = ""

    def payload(self) -> dict:
        return {
            "state": self.state,
            "question": self.question,
            "options": list(self.options),
            "label": self.label,
            "group": self.group,
            "family": self.family,
        }


def normalise(payload: dict) -> Row:
    state = payload.get("state", payload.get("context"))
    if not isinstance(state, str) or not state:
        raise ValueError("each row needs a non-empty string state (or context)")
    question = payload.get("question", "")
    if not isinstance(question, str):
        raise ValueError("question must be a string")

    raw = payload.get("options")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("options must be a list of at least two entries")
    options = []
    for entry in raw:
        if isinstance(entry, str):
            options.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("description"), str):
            options.append(entry["description"])
        else:
            raise ValueError("options must be strings or objects with a description")
    if any(not option for option in options):
        raise ValueError("options must be non-empty")

    label = payload.get("label")
    if not isinstance(label, int) or not 0 <= label < len(options):
        raise ValueError("label must be an index into options")
    return Row(
        state=state,
        question=question,
        options=tuple(options),
        label=label,
        group=str(payload.get("group", "")),
        family=str(payload.get("family", "")),
    )


def read_jsonl(path: str | Path) -> list[Row]:
    rows = [
        normalise(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def write_jsonl(path: str | Path, rows: list[Row]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.payload(), ensure_ascii=False) + "\n")
    return path


# ----------------------------------------------------------------- batching


def clip(text: str, limit: int) -> bytes:
    """Truncate to `limit` bytes. The runtime refuses longer input rather than
    truncating silently, so every caller must clip first and say that it did."""
    return text.encode("utf-8", errors="replace")[:limit]


class Collator:
    def __init__(self, config: BlinkConfig, max_options: int = 16) -> None:
        self.config = config
        self.max_options = max_options

    def __call__(self, rows: list[Row]) -> dict:
        import torch

        config = self.config
        states = [clip(row.state, config.max_state) for row in rows]
        questions = [clip(row.question, config.max_question) for row in rows]
        options = [
            [clip(option, config.max_option) or b" " for option in row.options]
            for row in rows
        ]
        if any(len(entry) > self.max_options for entry in options):
            raise ValueError("a row has more options than max_options")

        batch = len(rows)
        state_width = max(1, max(len(item) for item in states))
        question_width = max(1, max(len(item) for item in questions))
        option_count = max(len(entry) for entry in options)
        option_width = max(1, max(len(item) for entry in options for item in entry))

        def sequence(items: list[bytes], width: int):
            ids = np.zeros((len(items), width), dtype=np.uint8)
            mask = np.zeros((len(items), width), dtype=bool)
            for index, item in enumerate(items):
                ids[index, :len(item)] = np.frombuffer(item, dtype=np.uint8)
                mask[index, :len(item)] = True
            return ids, mask

        state_ids, state_mask = sequence(states, state_width)
        question_ids, question_mask = sequence(questions, question_width)

        option_ids = np.zeros((batch, option_count, option_width), dtype=np.uint8)
        option_mask = np.zeros((batch, option_count, option_width), dtype=bool)
        option_present = np.zeros((batch, option_count), dtype=bool)
        for r, entry in enumerate(options):
            for c, item in enumerate(entry):
                option_ids[r, c, :len(item)] = np.frombuffer(item, dtype=np.uint8)
                option_mask[r, c, :len(item)] = True
                option_present[r, c] = True

        return {
            "state_ids": torch.from_numpy(state_ids),
            "state_mask": torch.from_numpy(state_mask),
            "question_ids": torch.from_numpy(question_ids),
            "question_mask": torch.from_numpy(question_mask),
            "option_ids": torch.from_numpy(option_ids),
            "option_mask": torch.from_numpy(option_mask),
            "option_present": torch.from_numpy(option_present),
            "labels": torch.tensor([row.label for row in rows], dtype=torch.long),
        }


class RowDataset:
    """Minimal Dataset so torch.utils.data.DataLoader can shuffle rows."""

    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Row:
        return self.rows[index]


# ------------------------------------------------------- synthetic corpus

QUEUES = {
    "billing": ("invoice", "charge", "refund", "payment", "subscription price"),
    "access": ("password", "login", "two-factor code", "locked out", "sign-in"),
    "shipping": ("parcel", "tracking number", "courier", "delivery date", "package"),
    "hardware": ("battery", "screen", "keyboard", "fan noise", "charging port"),
    "privacy": ("data export", "account deletion", "consent record", "retention",
                "personal data"),
}
QUEUE_LABELS = {
    "billing": "billing and payments",
    "access": "account access and sign-in",
    "shipping": "delivery and logistics",
    "hardware": "device hardware repair",
    "privacy": "privacy and data rights",
}

# Entity pools are split three ways, not two. The test split must not reuse a
# training entity -- that much is obvious -- but the *validation* split must
# not either, because the calibration temperature is fitted on validation and a
# temperature fitted in-distribution does not survive the shift to held-out
# entities. An earlier two-way split produced a model with 0.69 accuracy and an
# ECE of 0.13 for exactly that reason.
#
# The three pools are the same size and shape, so option counts and chance
# levels match across splits and the baselines stay comparable.
CITIES = ("Turin", "Lisbon", "Bergen", "Osaka", "Quito", "Perth", "Tallinn",
          "Cusco", "Gdansk", "Nagoya", "Aarhus", "Rennes", "Utrecht", "Kanpur",
          "Salerno", "Tromso", "Ghent", "Malmo", "Pescara", "Leuven",
          "Brescia", "Tampere", "Girona", "Kaunas")
PRODUCTS = ("router", "thermostat", "doorbell", "scanner", "projector",
            "speaker", "handset", "printer", "modem", "camera", "tablet",
            "headset", "monitor", "keypad", "sensor", "charger", "adapter",
            "receiver")
NAMES = ("Iris", "Ravi", "Noor", "Milo", "Ana", "Teo", "Lena", "Kofi",
         "Hana", "Bo", "Ines", "Yuki", "Dara", "Omar", "Suvi", "Nils",
         "Mira", "Jonas", "Leila", "Pavel", "Rosa", "Aki", "Femi", "Elsa")

# Shared across the splits on purpose, like the queue labels: these are the
# vocabulary of the *task*, not of the entities. Splitting them would change
# what the model is asked to do rather than which entities it has seen.
VERBS = ("shipped", "cancelled", "refunded", "escalated", "approved", "rejected")

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Vocabulary:
    cities: tuple[str, ...]
    products: tuple[str, ...]
    names: tuple[str, ...]


def vocabulary(split: str = "train") -> Vocabulary:
    """The entity pool for one split. The three are pairwise disjoint."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    index = SPLITS.index(split)

    def slice_(pool: tuple[str, ...]) -> tuple[str, ...]:
        if len(pool) % 3:
            raise ValueError("entity pools must divide into three equal parts")
        size = len(pool) // 3
        return pool[index * size:(index + 1) * size]

    return Vocabulary(slice_(CITIES), slice_(PRODUCTS), slice_(NAMES))


def _pick(rng: random.Random, pool, count: int) -> list:
    return rng.sample(list(pool), count)


def _routing(rng: random.Random, vocab: Vocabulary) -> list[Row]:
    keys = _pick(rng, QUEUES, rng.randint(3, 5))
    target = rng.choice(keys)
    cue = rng.choice(QUEUES[target])
    name, city = rng.choice(vocab.names), rng.choice(vocab.cities)
    state = (
        f"Ticket from {name} in {city}. The customer writes about the {cue} "
        f"on their {rng.choice(vocab.products)} and asks for help today."
    )
    labels = [QUEUE_LABELS[key] for key in keys]
    rng.shuffle(labels)
    return [Row(state, "Which team should handle this ticket?", tuple(labels),
                labels.index(QUEUE_LABELS[target]), family="routing")]


JUDGMENT_OPTIONS = (
    "the evidence supports the claim",
    "the evidence is insufficient to decide",
    "the evidence contradicts the claim",
)


def _judgment(rng: random.Random, vocab: Vocabulary) -> list[Row]:
    name = rng.choice(vocab.names)
    product = rng.choice(vocab.products)
    verb = rng.choice(VERBS)
    other = rng.choice([v for v in VERBS if v != verb])
    state = f"The order for {name}'s {product} was {verb} on Tuesday morning."
    rows = []
    for claim, label in (
        (f"the order was {verb}", 0),
        (f"the order was {other}", 2),
        (f"the {product} arrived at {name}'s address", 1),
    ):
        options = list(JUDGMENT_OPTIONS)
        order = list(range(3))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        rows.append(Row(state, f"Assess the claim: {claim}.", tuple(shuffled),
                        order.index(label), family="judgment"))
    return rows


def _selection(rng: random.Random, vocab: Vocabulary) -> list[Row]:
    count = rng.randint(3, min(6, len(vocab.products)))
    people = _pick(rng, vocab.names, count)
    cities = _pick(rng, vocab.cities, count)
    products = _pick(rng, vocab.products, count)
    lines = [
        f"{person} in {city} owns a {product}."
        for person, city, product in zip(people, cities, products)
    ]
    rng.shuffle(lines)
    state = " ".join(lines)

    # The option list is shuffled independently of the sampling order, and the
    # people asked about are drawn at random. Without both, the correct answer
    # would sit at option index 0 or 1 on every row and "always answer the
    # first option" would beat the model. See test_no_family_leaks_the_label
    # _position, which exists because exactly that bug shipped once.
    options = list(cities)
    rng.shuffle(options)
    asked = rng.sample(range(count), min(2, count))
    return [
        Row(state, f"Which city does {people[index]} live in?",
            tuple(options), options.index(cities[index]), family="selection")
        for index in asked
    ]


def _comparison(rng: random.Random, vocab: Vocabulary) -> list[Row]:
    count = rng.randint(3, min(5, len(vocab.products)))
    products = _pick(rng, vocab.products, count)
    values = _pick(rng, range(11, 99), count)
    state = " ".join(
        f"The {product} scored {value} points in the review."
        for product, value in zip(products, values)
    )
    highest = products[int(np.argmax(values))]
    lowest = products[int(np.argmin(values))]
    options = list(products)
    rng.shuffle(options)
    return [
        Row(state, "Which product scored highest?", tuple(options),
            options.index(highest), family="comparison"),
        Row(state, "Which product scored lowest?", tuple(options),
            options.index(lowest), family="comparison"),
    ]


FAMILIES = (_routing, _judgment, _selection, _comparison)


def synthetic_rows(seed: int, groups: int, split: str = "train") -> list[Row]:
    """Generate `groups` independent scenario groups for one split.

    Every row produced by one group shares a group id. Splits are made on the
    group, never on the row, so that two questions about the same state can
    never land on opposite sides of the split. Each split also draws from its
    own disjoint entity pool; see `vocabulary`.
    """
    vocab = vocabulary(split)
    rows: list[Row] = []
    for index in range(groups):
        rng = random.Random(seed + index * 104_729)
        family = FAMILIES[index % len(FAMILIES)]
        group = f"g{seed}-{index}"
        for row in family(rng, vocab):
            rows.append(Row(row.state, row.question, row.options, row.label,
                            group=group, family=row.family))
    return rows


def write_synthetic(output: Path, groups: dict[str, int], seed: int) -> dict:
    """Write the splits. Each draws from its own disjoint entity pool."""
    counts = {}
    offset = 0
    for split, size in groups.items():
        rows = synthetic_rows(seed + offset, size, split=split)
        write_jsonl(output / f"{split}.jsonl", rows)
        counts[split] = len(rows)
        offset += 1_000_003
    return counts
