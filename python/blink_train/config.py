"""Model geometry. One dataclass drives the trainer, the exporter, the NumPy
reference and the C header, so the four can never drift apart silently."""

from __future__ import annotations

from dataclasses import asdict, dataclass

VOCAB = 256
SEGMENTS = 3
SEG_STATE, SEG_QUESTION, SEG_OPTION = 0, 1, 2


@dataclass(frozen=True)
class BlinkConfig:
    name: str = "blink-tiny"
    width: int = 64
    blocks: int = 1
    ffn_width: int = 128
    rank: int = 64
    heads: int = 2
    conv_width: int = 5
    stride: int = 4
    mixer_blocks: int = 1
    film: bool = True
    cross: bool = True
    bigram_buckets: int = 4096
    max_state: int = 256
    max_question: int = 128
    max_option: int = 48
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.width % 4:
            raise ValueError("width must be a multiple of four")
        if self.rank % self.heads:
            raise ValueError("rank must divide evenly into heads")
        if self.conv_width % 2 == 0 or self.conv_width > 31:
            raise ValueError("conv_width must be odd and at most 31")
        if not 1 <= self.stride <= 64:
            raise ValueError("stride must be between 1 and 64")
        if not 0 <= self.mixer_blocks <= self.blocks:
            raise ValueError("mixer_blocks must be between 0 and blocks")
        if (self.mixer_blocks or self.cross) and self.width % self.heads:
            raise ValueError("attention sub-layers need heads to divide width")
        if self.bigram_buckets & (self.bigram_buckets - 1):
            raise ValueError("bigram_buckets must be a power of two")
        if len(self.name.encode()) > 15:
            raise ValueError("name must fit in 15 bytes")
        for field in ("max_state", "max_question", "max_option"):
            if not 1 <= getattr(self, field) <= 4096:
                raise ValueError(f"{field} must be between 1 and 4096")

    @property
    def head_dim(self) -> int:
        return self.rank // self.heads

    def positions(self, length: int) -> int:
        """Pooled positions produced by `length` input bytes."""
        return (length + self.stride - 1) // self.stride

    @property
    def context_positions(self) -> int:
        return self.positions(self.max_state) + self.positions(self.max_question)

    def parameter_count(self) -> int:
        w, f = self.width, self.ffn_width
        stem = 2 * w + self.conv_width * w + w * w
        per_block = 2 * w + self.conv_width * w + w * w + 2 * w + 2 * f * w
        per_mixer = 2 * w + 4 * w * w
        per_film = (2 * w + 2 * w * w) if self.film else 0
        per_cross = (2 * w + 4 * w * w) if self.cross else 0
        return (
            VOCAB * w
            + self.bigram_buckets * w
            + (self.max_state + self.max_question + self.max_option) * w
            + SEGMENTS * w
            + stem
            + self.blocks * per_block
            + self.mixer_blocks * per_mixer
            + per_film
            + per_cross
            + 4 * w
            + 3 * self.rank * w
            + 1
        )

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, BlinkConfig] = {
    "tiny": BlinkConfig(
        name="blink-tiny",
        width=64,
        blocks=2,
        ffn_width=128,
        rank=64,
        heads=2,
        stride=4,
        mixer_blocks=1,
        bigram_buckets=4096,
        max_state=256,
        max_question=128,
        max_option=48,
    ),
    "small": BlinkConfig(
        name="blink-small",
        width=192,
        blocks=4,
        ffn_width=384,
        rank=192,
        heads=4,
        stride=8,
        mixer_blocks=2,
        bigram_buckets=32768,
        max_state=512,
        max_question=192,
        max_option=64,
    ),
    # Deliberately degenerate; used by the parity and container tests so they
    # stay fast and exercise odd shapes (single head, one-byte positions).
    "nano": BlinkConfig(
        name="blink-nano",
        width=16,
        blocks=2,
        ffn_width=32,
        rank=16,
        heads=1,
        conv_width=3,
        stride=2,
        mixer_blocks=1,
        bigram_buckets=64,
        max_state=48,
        max_question=24,
        max_option=12,
    ),
}


def preset(name: str) -> BlinkConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
