"""Export a trained PyTorch scorer to a .blink container.

The export is lossy in exactly one way: the tensors listed in `QUANTIZED` are
stored as symmetric per-row int8. `export_model` returns the measured
quantization error so the caller can record it rather than assume it is small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import BlinkConfig
from .container import ContainerWriter

# Tensors that ship as int8. Everything else stays fp32.
#
# The layer norms, the depthwise convolutions and the segment table are a few
# hundred values each: quantizing them costs accuracy for no measurable space.
#
# `film.gamma` and `film.beta` are larger but are excluded for a different
# reason. They scale the option vectors, so their quantization error is
# multiplicative, and a multiplicative error compounds through the head in a
# way an additive one does not. Quantizing them drove validation NLL from 0.87
# to 1.51 in a 100-epoch run; leaving them in fp32 costs 24 KiB at the tiny
# preset.
QUANTIZED = (
    "emb.unigram", "emb.bigram", "emb.pos_state", "emb.pos_question",
    "emb.pos_option", "stem.proj", "head.q", "head.k", "head.v",
)


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().to(dtype=__import__("torch").float32).numpy()


def collect_tensors(model) -> tuple[dict[str, np.ndarray], set[str]]:
    """Flatten a BlinkModel into container tensor names."""
    config: BlinkConfig = model.config
    tensors: dict[str, np.ndarray] = {
        "emb.unigram": _numpy(model.unigram.weight),
        "emb.bigram": _numpy(model.bigram.weight),
        "emb.pos_state": _numpy(model.pos_state.weight),
        "emb.pos_question": _numpy(model.pos_question.weight),
        "emb.pos_option": _numpy(model.pos_option.weight),
        "emb.segment": _numpy(model.segment),
        "stem.ln.gain": _numpy(model.stem.ln_gain),
        "stem.ln.bias": _numpy(model.stem.ln_bias),
        "stem.conv": _numpy(model.stem.conv),
        "stem.proj": _numpy(model.stem.proj.weight),
        "head.ctx_ln.gain": _numpy(model.ctx_ln_gain),
        "head.ctx_ln.bias": _numpy(model.ctx_ln_bias),
        "head.opt_ln.gain": _numpy(model.opt_ln_gain),
        "head.opt_ln.bias": _numpy(model.opt_ln_bias),
        "head.q": _numpy(model.wq.weight),
        "head.k": _numpy(model.wk.weight),
        "head.v": _numpy(model.wv.weight),
        "head.logit_scale": _numpy(model.logit_scale).reshape(1),
    }
    quantized = set(QUANTIZED)
    for index, block in enumerate(model.blocks):
        tensors[f"block{index}.ln1.gain"] = _numpy(block.ln1_gain)
        tensors[f"block{index}.ln1.bias"] = _numpy(block.ln1_bias)
        tensors[f"block{index}.conv"] = _numpy(block.conv)
        tensors[f"block{index}.glob"] = _numpy(block.glob.weight)
        tensors[f"block{index}.ln2.gain"] = _numpy(block.ln2_gain)
        tensors[f"block{index}.ln2.bias"] = _numpy(block.ln2_bias)
        tensors[f"block{index}.fc1"] = _numpy(block.fc1.weight)
        tensors[f"block{index}.fc2"] = _numpy(block.fc2.weight)
        quantized.update({
            f"block{index}.glob", f"block{index}.fc1", f"block{index}.fc2",
        })
        if block.mixer is not None:
            tensors[f"block{index}.mix.ln.gain"] = _numpy(block.mixer.ln_gain)
            tensors[f"block{index}.mix.ln.bias"] = _numpy(block.mixer.ln_bias)
            for part in ("q", "k", "v", "o"):
                name = f"block{index}.mix.{part}"
                tensors[name] = _numpy(getattr(block.mixer, part).weight)
                quantized.add(name)
    if config.cross:
        tensors["cross.ln.gain"] = _numpy(model.cross.ln_gain)
        tensors["cross.ln.bias"] = _numpy(model.cross.ln_bias)
        for part in ("q", "k", "v", "o"):
            name = f"cross.{part}"
            tensors[name] = _numpy(getattr(model.cross, part).weight)
            quantized.add(name)

    if config.film:
        tensors["film.ln.gain"] = _numpy(model.film_ln_gain)
        tensors["film.ln.bias"] = _numpy(model.film_ln_bias)
        tensors["film.gamma"] = _numpy(model.film_gamma)
        tensors["film.beta"] = _numpy(model.film_beta)
    return tensors, quantized


def export_model(model, path: str | Path, temperature: float = 1.0) -> dict:
    from dataclasses import replace

    config = replace(model.config, temperature=float(temperature))
    writer = ContainerWriter(config)
    tensors, quantized = collect_tensors(model)

    error = {}
    for name, array in tensors.items():
        if name in quantized:
            restored = writer.add_quantized(name, array)
            scale = float(np.abs(array).max()) or 1.0
            error[name] = float(np.abs(restored - array).max() / scale)
        else:
            writer.add_f32(name, array)

    written = writer.write(path)
    size = written.stat().st_size
    return {
        "path": str(written),
        "bytes": size,
        "parameters": config.parameter_count(),
        "bytes_per_parameter": size / config.parameter_count(),
        "temperature": config.temperature,
        "quantized_tensors": sorted(quantized),
        "max_relative_quantization_error": max(error.values()) if error else 0.0,
        "worst_tensor": max(error, key=error.get) if error else None,
    }
