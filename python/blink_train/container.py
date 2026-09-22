"""Reader and writer for the .blink container.

Layout (little-endian throughout):

    0    char[8]  magic "BLNKMDL\\0"
    8    u32      format version
    12   u32      flags (bit 0 = little-endian marker)
    16   u32      header bytes (128)
    20   u32      tensor count
    24   u64      blob offset (64-byte aligned)
    32   u64      blob bytes
    40   u32      crc32 of the whole file with this field read as zero
    44   u32      reserved
    48   ...      64 bytes of configuration: width, blocks, ffn_width, rank,
                  heads, conv_width, stride, bigram_buckets, max_state,
                  max_question, max_option, temperature (f32), vocab,
                  mixer_blocks, film, cross
    112  char[16] preset name
    128  ...      tensor table, 64 bytes per entry
    ...  ...      64-byte aligned tensor blob

A tensor entry is char[32] name, u32 dtype, u32 ndim, u32 dims[4], u64 offset.
Each int8 matrix is accompanied by a "<name>.scale" fp32 vector with one entry
per row.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .config import BlinkConfig

MAGIC = b"BLNKMDL\x00"
FORMAT_VERSION = 3  # see src/blink_internal.h for what changed
FLAG_LITTLE_ENDIAN = 1
HEADER_BYTES = 128
ENTRY_BYTES = 64
NAME_BYTES = 32
ALIGN = 64

DTYPE_F32 = 0
DTYPE_I8 = 1
_NUMPY = {DTYPE_F32: np.dtype("<f4"), DTYPE_I8: np.dtype("i1")}


def _align(value: int) -> int:
    return (value + ALIGN - 1) & ~(ALIGN - 1)


def quantize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8 quantization; returns (int8 matrix, fp32 scales).

    A row of zeros keeps a scale of 1.0 so the runtime never divides by zero
    and the round trip stays exact.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("quantize_rows expects a 2-D matrix")
    peak = np.abs(values).max(axis=1)
    scale = np.where(peak > 0, peak / 127.0, 1.0)
    quantized = np.rint(values / scale[:, None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return quantized, scale.astype(np.float32)


def dequantize_rows(quantized: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return quantized.astype(np.float64) * np.asarray(scale, dtype=np.float64)[:, None]


class ContainerWriter:
    """Collects tensors, then serialises them in one pass."""

    def __init__(self, config: BlinkConfig) -> None:
        self.config = config
        self._tensors: list[tuple[str, np.ndarray, int]] = []
        self._names: set[str] = set()

    def add_f32(self, name: str, array: np.ndarray) -> None:
        self._add(name, np.ascontiguousarray(array, dtype="<f4"), DTYPE_F32)

    def add_i8(self, name: str, array: np.ndarray) -> None:
        self._add(name, np.ascontiguousarray(array, dtype=np.int8), DTYPE_I8)

    def add_quantized(self, name: str, matrix: np.ndarray) -> np.ndarray:
        """Quantize a 2-D matrix, store it with its scales, return the
        dequantized values actually represented by the file."""
        quantized, scale = quantize_rows(matrix)
        self.add_i8(name, quantized)
        self.add_f32(f"{name}.scale", scale)
        return dequantize_rows(quantized, scale)

    def _add(self, name: str, array: np.ndarray, dtype: int) -> None:
        if len(name.encode()) >= NAME_BYTES:
            raise ValueError(f"tensor name too long: {name}")
        if name in self._names:
            raise ValueError(f"duplicate tensor name: {name}")
        if array.ndim not in (1, 2):
            raise ValueError("tensors must be 1-D or 2-D")
        self._names.add(name)
        self._tensors.append((name, array, dtype))

    def _header(self, tensor_count: int, blob_offset: int, blob_bytes: int,
                crc: int) -> bytes:
        c = self.config
        header = bytearray(HEADER_BYTES)
        header[0:8] = MAGIC
        struct.pack_into(
            "<IIII", header, 8, FORMAT_VERSION, FLAG_LITTLE_ENDIAN,
            HEADER_BYTES, tensor_count,
        )
        struct.pack_into("<QQ", header, 24, blob_offset, blob_bytes)
        struct.pack_into("<II", header, 40, crc, 0)
        struct.pack_into(
            "<IIIIIIIIIIIfI", header, 48,
            c.width, c.blocks, c.ffn_width, c.rank, c.heads, c.conv_width,
            c.stride, c.bigram_buckets, c.max_state, c.max_question,
            c.max_option, c.temperature, 256,
        )
        struct.pack_into("<III", header, 100, c.mixer_blocks, int(c.film),
                         int(c.cross))
        name = c.name.encode()[:15]
        header[112:112 + len(name)] = name
        return bytes(header)

    def build(self) -> bytes:
        table_bytes = len(self._tensors) * ENTRY_BYTES
        blob_offset = _align(HEADER_BYTES + table_bytes)

        table = bytearray()
        blob = bytearray()
        for name, array, dtype in self._tensors:
            offset = _align(len(blob))
            blob.extend(b"\x00" * (offset - len(blob)))
            blob.extend(array.tobytes(order="C"))
            dims = list(array.shape) + [0] * (4 - array.ndim)
            entry = bytearray(ENTRY_BYTES)
            encoded = name.encode()
            entry[0:len(encoded)] = encoded
            struct.pack_into("<II", entry, 32, dtype, array.ndim)
            struct.pack_into("<IIII", entry, 40, *dims)
            struct.pack_into("<Q", entry, 56, offset)
            table.extend(entry)

        body = (
            self._header(len(self._tensors), blob_offset, len(blob), 0)
            + bytes(table)
            + b"\x00" * (blob_offset - HEADER_BYTES - table_bytes)
            + bytes(blob)
        )
        crc = zlib.crc32(body[:40] + b"\x00\x00\x00\x00" + body[44:]) & 0xFFFFFFFF
        return (
            self._header(len(self._tensors), blob_offset, len(blob), crc)
            + body[HEADER_BYTES:]
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.build())
        return path


class Container:
    """Read-only view over a .blink file, used by the reference and the tests."""

    def __init__(self, data: bytes, verify: bool = True) -> None:
        if data[:8] != MAGIC:
            raise ValueError("not a .blink container")
        (version, flags, header_bytes, count) = struct.unpack_from("<IIII", data, 8)
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported container version {version}")
        if not flags & FLAG_LITTLE_ENDIAN:
            raise ValueError("container is not little-endian")
        blob_offset, blob_bytes = struct.unpack_from("<QQ", data, 24)
        stored_crc = struct.unpack_from("<I", data, 40)[0]
        if verify:
            actual = zlib.crc32(data[:40] + b"\x00\x00\x00\x00" + data[44:])
            if actual & 0xFFFFFFFF != stored_crc:
                raise ValueError("container checksum mismatch")

        fields = struct.unpack_from("<IIIIIIIIIIIfI", data, 48)
        self.config = BlinkConfig(
            name=data[112:128].split(b"\x00")[0].decode(),
            width=fields[0], blocks=fields[1], ffn_width=fields[2],
            rank=fields[3], heads=fields[4], conv_width=fields[5],
            stride=fields[6], bigram_buckets=fields[7], max_state=fields[8],
            max_question=fields[9], max_option=fields[10],
            temperature=fields[11],
            mixer_blocks=struct.unpack_from("<I", data, 100)[0],
            film=bool(struct.unpack_from("<I", data, 104)[0]),
            cross=bool(struct.unpack_from("<I", data, 108)[0]),
        )
        self.blob_bytes = blob_bytes
        self.tensors: dict[str, np.ndarray] = {}
        for index in range(count):
            base = header_bytes + index * ENTRY_BYTES
            name = data[base:base + NAME_BYTES].split(b"\x00")[0].decode()
            dtype, ndim = struct.unpack_from("<II", data, base + 32)
            dims = struct.unpack_from("<IIII", data, base + 40)[:ndim]
            offset = struct.unpack_from("<Q", data, base + 56)[0]
            numpy_dtype = _NUMPY[dtype]
            start = blob_offset + offset
            size = int(np.prod(dims)) * numpy_dtype.itemsize
            array = np.frombuffer(data, dtype=numpy_dtype, count=int(np.prod(dims)),
                                  offset=start)
            self.tensors[name] = array.reshape(dims)
            del size

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> "Container":
        return cls(Path(path).read_bytes(), verify=verify)

    def matrix(self, name: str) -> np.ndarray:
        """Dequantized float64 view of a tensor, int8 or fp32 alike."""
        array = self.tensors[name]
        if array.dtype == np.int8:
            return dequantize_rows(array, self.tensors[f"{name}.scale"])
        return array.astype(np.float64)

    def vector(self, name: str) -> np.ndarray:
        return self.tensors[name].astype(np.float64)
