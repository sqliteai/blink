"""Blink training, export and reference tooling.

The C runtime in src/ is the product; this package builds the weights it runs
and provides the float64 reference the runtime is tested against.
"""

from .config import PRESETS, BlinkConfig, preset
from .container import Container, ContainerWriter, dequantize_rows, quantize_rows
from .data import Collator, Row, normalise, read_jsonl, write_jsonl
from .hashing import bigram_index, bigram_indices
from .reference import ReferenceModel

__all__ = [
    "BlinkConfig",
    "Collator",
    "Container",
    "ContainerWriter",
    "PRESETS",
    "ReferenceModel",
    "Row",
    "bigram_index",
    "bigram_indices",
    "dequantize_rows",
    "normalise",
    "preset",
    "quantize_rows",
    "read_jsonl",
    "write_jsonl",
]
__version__ = "0.1.0"
