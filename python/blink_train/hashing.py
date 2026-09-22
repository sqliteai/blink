"""Byte-bigram hashing.

This must produce the same bucket as `blink_bigram_index` in
src/blink_kernels.c for every one of the 65,536 byte pairs. tests/python
checks that exhaustively against the C implementation.
"""

from __future__ import annotations

import numpy as np

MASK32 = 0xFFFFFFFF


def bigram_index(previous: int, current: int, buckets: int) -> int:
    h = (previous * 0x9E3779B1) & MASK32
    h ^= (current * 0x85EBCA77) & MASK32
    h ^= h >> 15
    h = (h * 0x2545F491) & MASK32
    h ^= h >> 13
    return h & (buckets - 1)


def bigram_table(buckets: int) -> np.ndarray:
    """All 256x256 bucket indices, indexed as table[previous, current]."""
    previous = np.arange(256, dtype=np.uint64)[:, None]
    current = np.arange(256, dtype=np.uint64)[None, :]
    h = (previous * np.uint64(0x9E3779B1)) & np.uint64(MASK32)
    h = h ^ ((current * np.uint64(0x85EBCA77)) & np.uint64(MASK32))
    h = h ^ (h >> np.uint64(15))
    h = (h * np.uint64(0x2545F491)) & np.uint64(MASK32)
    h = h ^ (h >> np.uint64(13))
    return (h & np.uint64(buckets - 1)).astype(np.int64)


def bigram_indices(data: np.ndarray, buckets: int) -> np.ndarray:
    """Bucket index per position for a [batch, length] uint8 array.

    Position 0 uses a zero sentinel as its predecessor, matching the runtime.
    """
    table = bigram_table(buckets)
    previous = np.zeros_like(data)
    previous[:, 1:] = data[:, :-1]
    return table[previous, data]
