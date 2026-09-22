"""Float64 reference implementation of the Blink forward pass.

This module is the arbiter of correctness. It reads a .blink container, applies
exactly the operations that src/blink_runtime.c applies, and does so in double
precision so that any disagreement with the C runtime is attributable to fp32
rounding rather than to a difference in the definition.

It is deliberately written as plain loops over a single decision: readability
and a literal correspondence with the C source matter more here than speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SEG_OPTION, SEG_QUESTION, SEG_STATE
from .container import Container
from .hashing import bigram_index


def layer_norm(x: np.ndarray, gain: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean()
    variance = ((x - mean) ** 2).mean()
    return (x - mean) / np.sqrt(variance + 1e-5) * gain + bias


def relu2(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x * x, 0.0)


def quantize_activations(x: np.ndarray) -> np.ndarray:
    """The W8A8 build's activation rounding, row by row: a symmetric int8
    scale per row, round half to even, back to real values. Mirrors
    blink_quantize_row in src/blink_kernels.c."""
    x = np.atleast_2d(x)
    out = np.zeros_like(x)
    for i, row in enumerate(x):
        maximum = np.abs(row).max() if row.size else 0.0
        if maximum > 0:
            q = np.clip(np.rint(row * (127.0 / maximum)), -127, 127)
            out[i] = q * (maximum / 127.0)
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = np.exp(x - x.max())
    return shifted / shifted.sum()


@dataclass
class Decision:
    probabilities: np.ndarray
    logits: np.ndarray
    attention: np.ndarray  # [options, context] averaged over heads


class ReferenceModel:
    def __init__(self, container: Container, activations: str = "fp32") -> None:
        """`activations="int8"` reproduces the W8A8 build (`make W8A8=1`):
        the input of every int8 projection is quantized first. The default
        is the normative W8A32 forward pass."""
        if activations not in ("fp32", "int8"):
            raise ValueError(f"unknown activations {activations!r}")
        self.activations = activations
        self.container = container
        self.config = container.config
        self.unigram = container.matrix("emb.unigram")
        self.bigram = container.matrix("emb.bigram")
        self.positions = {
            SEG_STATE: container.matrix("emb.pos_state"),
            SEG_QUESTION: container.matrix("emb.pos_question"),
            SEG_OPTION: container.matrix("emb.pos_option"),
        }
        self.segment = container.matrix("emb.segment")
        self.stem = {
            "ln_gain": container.vector("stem.ln.gain"),
            "ln_bias": container.vector("stem.ln.bias"),
            "conv": container.matrix("stem.conv"),
            "proj": container.matrix("stem.proj"),
        }
        self.blocks = [
            {
                "ln1_gain": container.vector(f"block{i}.ln1.gain"),
                "ln1_bias": container.vector(f"block{i}.ln1.bias"),
                "conv": container.matrix(f"block{i}.conv"),
                "glob": container.matrix(f"block{i}.glob"),
                "ln2_gain": container.vector(f"block{i}.ln2.gain"),
                "ln2_bias": container.vector(f"block{i}.ln2.bias"),
                "fc1": container.matrix(f"block{i}.fc1"),
                "fc2": container.matrix(f"block{i}.fc2"),
                "mixer": (
                    {
                        "ln_gain": container.vector(f"block{i}.mix.ln.gain"),
                        "ln_bias": container.vector(f"block{i}.mix.ln.bias"),
                        "q": container.matrix(f"block{i}.mix.q"),
                        "k": container.matrix(f"block{i}.mix.k"),
                        "v": container.matrix(f"block{i}.mix.v"),
                        "o": container.matrix(f"block{i}.mix.o"),
                    }
                    if i >= self.config.blocks - self.config.mixer_blocks
                    else None
                ),
            }
            for i in range(self.config.blocks)
        ]
        self.ctx_ln = (container.vector("head.ctx_ln.gain"),
                       container.vector("head.ctx_ln.bias"))
        self.opt_ln = (container.vector("head.opt_ln.gain"),
                       container.vector("head.opt_ln.bias"))
        self.cross = (
            {
                "ln_gain": container.vector("cross.ln.gain"),
                "ln_bias": container.vector("cross.ln.bias"),
                "q": container.matrix("cross.q"),
                "k": container.matrix("cross.k"),
                "v": container.matrix("cross.v"),
                "o": container.matrix("cross.o"),
            }
            if self.config.cross
            else None
        )
        self.film = (
            {
                "ln_gain": container.vector("film.ln.gain"),
                "ln_bias": container.vector("film.ln.bias"),
                "gamma": container.matrix("film.gamma"),
                "beta": container.matrix("film.beta"),
            }
            if self.config.film
            else None
        )
        self.wq = container.matrix("head.q")
        self.wk = container.matrix("head.k")
        self.wv = container.matrix("head.v")
        self.logit_scale = float(container.vector("head.logit_scale")[0])

    @classmethod
    def load(cls, path: str | Path, verify: bool = True,
             activations: str = "fp32") -> "ReferenceModel":
        return cls(Container.load(path, verify=verify), activations)

    def _proj(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """x @ weight.T for an int8 projection, one row of x at a time."""
        if self.activations == "int8":
            product = quantize_activations(x) @ weight.T
            return product[0] if np.ndim(x) == 1 else product
        return x @ weight.T

    # ------------------------------------------------------------- encoder

    def _attend(self, x: np.ndarray, weights: dict, keys: np.ndarray,
                values: np.ndarray, normalise_x: bool = True) -> np.ndarray:
        """One attention sub-layer: queries from `x`, keys and values given."""
        config = self.config
        head_dim = config.width // config.heads
        inverse = 1.0 / np.sqrt(head_dim)
        a = np.stack([layer_norm(row, weights["ln_gain"], weights["ln_bias"])
                      for row in x]) if normalise_x else x
        queries = self._proj(a, weights["q"])
        attended = np.zeros_like(queries)
        for t in range(len(x)):
            for h in range(config.heads):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                scores = softmax(keys[:, sl] @ queries[t, sl] * inverse)
                attended[t, sl] = scores @ values[:, sl]
        return x + self._proj(attended, weights["o"])

    def cross_kv(self, state: bytes) -> tuple[np.ndarray, np.ndarray] | None:
        """The state side of the cross layer, computed once per state."""
        if self.cross is None or not state:
            return None
        encoded = self.encode(state, SEG_STATE)
        a = np.stack([layer_norm(row, self.cross["ln_gain"],
                                 self.cross["ln_bias"]) for row in encoded])
        return self._proj(a, self.cross["k"]), self._proj(a, self.cross["v"])

    def encode(self, data: bytes, segment: int, cross=None) -> np.ndarray:
        config = self.config
        length = len(data)
        if length == 0:
            return np.zeros((0, config.width))
        positions = self.positions[segment]
        x = np.zeros((length, config.width))
        for t in range(length):
            current = data[t]
            previous = data[t - 1] if t else 0
            bucket = bigram_index(previous, current, config.bigram_buckets)
            x[t] = (
                self.unigram[current]
                + self.bigram[bucket]
                + positions[t]
                + self.segment[segment]
            )

        half = config.conv_width // 2

        # stem at full byte resolution
        a = np.stack([layer_norm(x[t], self.stem["ln_gain"], self.stem["ln_bias"])
                      for t in range(length)])
        local = np.zeros_like(x)
        for t in range(length):
            for k in range(config.conv_width):
                source = t + k - half
                if 0 <= source < length:
                    local[t] += self.stem["conv"][k] * a[source]
        x = x + local + self._proj(a, self.stem["proj"])

        # pool, then the blocks at the reduced resolution
        stride = config.stride
        if stride > 1:
            positions = (length + stride - 1) // stride
            x = np.stack([x[p * stride:min((p + 1) * stride, length)].mean(axis=0)
                          for p in range(positions)])
            length = positions

        for block in self.blocks:
            a = np.stack(
                [layer_norm(x[t], block["ln1_gain"], block["ln1_bias"])
                 for t in range(length)]
            )
            global_term = self._proj(a.mean(axis=0), block["glob"])
            local = np.zeros_like(x)
            for t in range(length):
                for k in range(config.conv_width):
                    source = t + k - half
                    if 0 <= source < length:
                        local[t] += block["conv"][k] * a[source]
            x = x + local + global_term

            b = np.stack(
                [layer_norm(x[t], block["ln2_gain"], block["ln2_bias"])
                 for t in range(length)]
            )
            x = x + self._proj(relu2(self._proj(b, block["fc1"])), block["fc2"])
            if block["mixer"] is not None:
                x = self._mix(x, block["mixer"], length)

        if self.cross is not None and segment == SEG_QUESTION and cross is not None:
            x = self._attend(x, self.cross, cross[0], cross[1])
        return x

    def _mix(self, x: np.ndarray, mixer: dict, length: int) -> np.ndarray:
        del length
        a = np.stack([layer_norm(row, mixer["ln_gain"], mixer["ln_bias"])
                      for row in x])
        return self._attend(x, mixer, self._proj(a, mixer["k"]),
                            self._proj(a, mixer["v"]))

    # ---------------------------------------------------------------- head

    def context_kv(self, state: bytes, question: bytes):
        encoded_question = self.encode(question, SEG_QUESTION,
                                       cross=self.cross_kv(state))
        hidden = np.concatenate(
            [self.encode(state, SEG_STATE), encoded_question], axis=0)
        normed = np.stack([layer_norm(row, *self.ctx_ln) for row in hidden])
        summary = (encoded_question.mean(axis=0) if len(encoded_question)
                   else None)
        return self._proj(normed, self.wk), self._proj(normed, self.wv), summary

    def option_queries(self, options: list[bytes],
                       summary: np.ndarray | None) -> np.ndarray:
        gamma = np.zeros(self.config.width)
        beta = np.zeros(self.config.width)
        if self.film is not None and summary is not None:
            normed = layer_norm(summary, self.film["ln_gain"], self.film["ln_bias"])
            gamma = self.film["gamma"] @ normed
            beta = self.film["beta"] @ normed
        queries = []
        for option in options:
            vector = self.encode(option, SEG_OPTION).mean(axis=0)
            modulated = vector * (1.0 + gamma) + beta
            queries.append(self._proj(layer_norm(modulated, *self.opt_ln), self.wq))
        return np.stack(queries)

    def decide(self, state: bytes, question: bytes, options: list[bytes]) -> Decision:
        config = self.config
        keys, values, summary = self.context_kv(state, question)
        queries = self.option_queries(options, summary)
        context = keys.shape[0]
        if context == 0:
            raise ValueError("a decision needs at least one context byte")

        head_dim = config.head_dim
        span = np.sqrt(head_dim)
        logits = np.zeros(len(options))
        attention = np.zeros((len(options), context))

        def unit(x: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(x, axis=-1, keepdims=True)
            return np.where(norm > 0, x / np.where(norm > 0, norm, 1.0), x)

        for n in range(len(options)):
            for h in range(config.heads):
                slice_ = slice(h * head_dim, (h + 1) * head_dim)
                query = unit(queries[n, slice_])
                scores = softmax(unit(keys[:, slice_]) @ query * span)
                attended = unit(scores @ values[:, slice_])
                logits[n] += float(query @ attended)
                attention[n] += scores
            attention[n] /= config.heads
        logits *= self.logit_scale
        return Decision(
            probabilities=softmax(logits / config.temperature),
            logits=logits,
            attention=attention,
        )
