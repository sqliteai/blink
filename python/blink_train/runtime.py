"""ctypes binding to the Blink C runtime.

Everything that reports accuracy or latency in this repository goes through
this binding, so the published numbers describe the C runtime that ships, not
the PyTorch model it was trained from.
"""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SUFFIX = "dylib" if platform.system() == "Darwin" else "so"
# BLINK_LIBRARY points the harness at another build of the same library, for
# example build-accelerate/libblink.dylib from `make ACCELERATE=1`.
DEFAULT_LIBRARY = Path(os.environ.get(
    "BLINK_LIBRARY", REPO_ROOT / "build" / f"libblink.{_SUFFIX}"))

BLINK_OK = 0
_STATUS = {
    0: "ok", 1: "invalid argument", 2: "io error", 3: "malformed container",
    4: "unsupported container version", 5: "checksum mismatch",
    6: "arena too small", 7: "input exceeds session limits",
    8: "state or menu not set",
}


class BlinkError(RuntimeError):
    def __init__(self, status: int, where: str) -> None:
        super().__init__(f"{where}: {_STATUS.get(status, 'unknown')} ({status})")
        self.status = status


class ModelInfo(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32), ("blocks", ctypes.c_uint32),
        ("ffn_width", ctypes.c_uint32), ("rank", ctypes.c_uint32),
        ("heads", ctypes.c_uint32), ("conv_width", ctypes.c_uint32),
        ("stride", ctypes.c_uint32), ("mixer_blocks", ctypes.c_uint32),
        ("film", ctypes.c_uint32), ("cross", ctypes.c_uint32),
        ("bigram_buckets", ctypes.c_uint32), ("max_state", ctypes.c_uint32),
        ("max_question", ctypes.c_uint32), ("max_option", ctypes.c_uint32),
        ("temperature", ctypes.c_float), ("parameters", ctypes.c_uint64),
        ("weights_bytes", ctypes.c_uint64), ("name", ctypes.c_char * 32),
    ]


class Limits(ctypes.Structure):
    _fields_ = [
        ("max_state", ctypes.c_uint32), ("max_question", ctypes.c_uint32),
        ("max_options", ctypes.c_uint32), ("max_option", ctypes.c_uint32),
    ]


class Result(ctypes.Structure):
    _fields_ = [
        ("options", ctypes.c_uint32), ("argmax", ctypes.c_uint32),
        ("confidence", ctypes.c_float), ("entropy", ctypes.c_float),
        ("margin", ctypes.c_float), ("state_bytes", ctypes.c_uint32),
        ("question_bytes", ctypes.c_uint32),
        ("context_positions", ctypes.c_uint32),
        ("encode_seconds", ctypes.c_double), ("head_seconds", ctypes.c_double),
    ]


def _bind(library: ctypes.CDLL) -> ctypes.CDLL:
    c, p, u32, f32 = ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_float
    size_t, status_p = ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)

    library.blink_version.restype = c
    library.blink_backend.restype = c
    library.blink_status_string.argtypes = [ctypes.c_int]
    library.blink_status_string.restype = c
    library.blink_model_open_file.argtypes = [c, ctypes.c_int, status_p]
    library.blink_model_open_file.restype = p
    library.blink_model_close.argtypes = [p]
    library.blink_model_get_info.argtypes = [p, ctypes.POINTER(ModelInfo)]
    library.blink_session_size.argtypes = [p, ctypes.POINTER(Limits)]
    library.blink_session_size.restype = size_t
    library.blink_session_create.argtypes = [p, ctypes.POINTER(Limits), status_p]
    library.blink_session_create.restype = p
    library.blink_session_free.argtypes = [p]
    library.blink_state_set.argtypes = [p, c, size_t]
    library.blink_state_set.restype = ctypes.c_int
    library.blink_menu_set.argtypes = [p, ctypes.POINTER(c), ctypes.POINTER(size_t), u32]
    library.blink_menu_set.restype = ctypes.c_int
    library.blink_score.argtypes = [p, c, size_t, ctypes.POINTER(f32),
                                    ctypes.POINTER(Result)]
    library.blink_score.restype = ctypes.c_int
    library.blink_last_logits.argtypes = [p, ctypes.POINTER(u32)]
    library.blink_last_logits.restype = ctypes.POINTER(f32)
    library.blink_last_attention.argtypes = [p, u32, ctypes.POINTER(f32), u32]
    library.blink_last_attention.restype = u32
    return library


_cache: dict[str, ctypes.CDLL] = {}


def load_library(path: str | Path | None = None) -> ctypes.CDLL:
    resolved = str(Path(path or DEFAULT_LIBRARY).resolve())
    if resolved not in _cache:
        if not Path(resolved).exists():
            raise FileNotFoundError(
                f"{resolved} not found; run `make` to build the runtime"
            )
        _cache[resolved] = _bind(ctypes.CDLL(resolved))
    return _cache[resolved]


def backend(library: str | Path | None = None) -> str:
    """The numeric backend of the loaded library, as blink_backend reports it:
    "neon", "scalar", "accelerate", or "w8a8-..."."""
    return load_library(library).blink_backend().decode()


class Model:
    def __init__(self, path: str | Path, verify: bool = True,
                 library: str | Path | None = None) -> None:
        self.library = load_library(library)
        status = ctypes.c_int(0)
        handle = self.library.blink_model_open_file(
            str(path).encode(), 1 if verify else 0, ctypes.byref(status)
        )
        if not handle:
            raise BlinkError(status.value, f"open {path}")
        self._handle = handle
        self.info = ModelInfo()
        self.library.blink_model_get_info(handle, ctypes.byref(self.info))

    @property
    def handle(self):
        return self._handle

    def session_size(self, limits: Limits | None = None) -> int:
        return int(self.library.blink_session_size(
            self._handle, ctypes.byref(limits) if limits else None
        ))

    def session(self, **kwargs) -> "Session":
        return Session(self, **kwargs)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.blink_model_close(self._handle)
            self._handle = None

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class Session:
    def __init__(self, model: Model, max_state: int = 0, max_question: int = 0,
                 max_options: int = 16, max_option: int = 0) -> None:
        self.model = model
        self.library = model.library
        self.limits = Limits(max_state, max_question, max_options, max_option)
        self.arena_bytes = model.session_size(self.limits)
        status = ctypes.c_int(0)
        handle = self.library.blink_session_create(
            model.handle, ctypes.byref(self.limits), ctypes.byref(status)
        )
        if not handle:
            raise BlinkError(status.value, "session_create")
        self._handle = handle
        self._menu_size = 0
        self._probabilities = (ctypes.c_float * 64)()

    def set_state(self, state: bytes | str) -> None:
        data = state.encode() if isinstance(state, str) else state
        code = self.library.blink_state_set(self._handle, data, len(data))
        if code != BLINK_OK:
            raise BlinkError(code, "state_set")

    def set_menu(self, options: list[bytes | str]) -> None:
        encoded = [o.encode() if isinstance(o, str) else o for o in options]
        count = len(encoded)
        array = (ctypes.c_char_p * count)(*encoded)
        lengths = (ctypes.c_size_t * count)(*[len(o) for o in encoded])
        code = self.library.blink_menu_set(self._handle, array, lengths, count)
        if code != BLINK_OK:
            raise BlinkError(code, "menu_set")
        self._menu_size = count

    def score(self, question: bytes | str = b"") -> tuple[list[float], Result]:
        data = question.encode() if isinstance(question, str) else question
        result = Result()
        code = self.library.blink_score(self._handle, data, len(data),
                                        self._probabilities, ctypes.byref(result))
        if code != BLINK_OK:
            raise BlinkError(code, "score")
        return list(self._probabilities[:self._menu_size]), result

    def decide(self, state, question, options) -> tuple[list[float], Result]:
        self.set_state(state)
        self.set_menu(options)
        return self.score(question)

    def logits(self) -> list[float]:
        count = ctypes.c_uint32(0)
        pointer = self.library.blink_last_logits(self._handle, ctypes.byref(count))
        if not pointer:
            raise BlinkError(8, "last_logits")
        return [pointer[i] for i in range(count.value)]

    def attention(self, option: int, capacity: int = 4096) -> list[float]:
        buffer = (ctypes.c_float * capacity)()
        written = self.library.blink_last_attention(self._handle, option, buffer,
                                                    capacity)
        return list(buffer[:written])

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.blink_session_free(self._handle)
            self._handle = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
