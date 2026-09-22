"""C runtime against the float64 reference, and the reference against PyTorch.

Three claims are checked here, and they are what makes every other number in
this repository trustworthy:

1. The C runtime computes the same function as `blink_train.reference`, which
   is written as plain float64 loops mirroring src/blink_runtime.c term by
   term. Disagreement beyond fp32 rounding means one of them is wrong.
2. The PyTorch training graph computes the same function as the reference,
   once the exported int8 weights are read back. Disagreement means the model
   that ships is not the model that was trained.
3. The bigram hash agrees across C, NumPy and pure Python for all 65,536 byte
   pairs, so an exported embedding table is indexed identically everywhere.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from blink_train.config import PRESETS, preset  # noqa: E402
from blink_train.container import Container, dequantize_rows, quantize_rows  # noqa: E402
from blink_train.data import Collator, Row  # noqa: E402
from blink_train.export import export_model  # noqa: E402
from blink_train.hashing import bigram_index, bigram_table  # noqa: E402
from blink_train.model import BlinkModel  # noqa: E402
from blink_train.reference import ReferenceModel  # noqa: E402
from blink_train.runtime import Model, backend, load_library  # noqa: E402

# fp32 accumulation against float64 accumulation. Anything larger than this
# means a real difference in the computation, not rounding.
PROBABILITY_TOLERANCE = 1e-5
LOGIT_TOLERANCE = 1e-4

# The W8A8 build (`make W8A8=1`, pointed at with BLINK_LIBRARY) is compared
# with the reference in its int8-activation mode. A value that lands within
# fp32 rounding of a quantization boundary can round to the neighbouring step
# in one implementation and not the other, so agreement is looser: observed
# at most 3.5e-4 on probabilities, against about 3e-3 for the same build
# compared with the fp32 reference, which is the quantization itself.
try:
    W8A8 = backend().startswith("w8a8")
except FileNotFoundError:
    W8A8 = False
ACTIVATIONS = "int8" if W8A8 else "fp32"
if W8A8:
    PROBABILITY_TOLERANCE = 1e-3
    LOGIT_TOLERANCE = 1e-2

CASES = [
    (b"The parcel left the depot on Monday and has not arrived.",
     b"Which team should handle this ticket?",
     [b"delivery and logistics", b"billing and payments", b"privacy"]),
    (b"x", b"", [b"yes", b"no"]),
    (b"", b"Which one?", [b"alpha", b"beta"]),
    (bytes(range(256)), b"\x00\x01\x02", [b"\xff\xfe", b"ok", b"third option"]),
    (b"a" * 61, b"b" * 23, [b"c" * 11, b"d", b"ef", b"ghij"]),
    ("caffè è ✓ \U0001F680".encode(), "perché?".encode(),
     ["sì".encode(), "no".encode(), "forse".encode()]),
]


def fit(data: bytes, limit: int) -> bytes:
    return data[:limit]


@pytest.fixture(scope="session")
def library():
    path = ROOT / "build"
    if not any(path.glob("libblink.*")):
        pytest.skip("run `make` first")
    return load_library()


@pytest.fixture(scope="session", params=sorted(PRESETS))
def exported(request, tmp_path_factory, library):
    """A freshly initialised model of each preset, exported to a container."""
    del library
    torch.manual_seed(1234 + len(request.param))
    config = preset(request.param)
    model = BlinkModel(config)
    # Random init leaves several tensors at zero; nudge everything so the test
    # exercises real values rather than a degenerate all-zero forward pass.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    path = tmp_path_factory.mktemp("blink") / f"{request.param}.blink"
    info = export_model(model, path, temperature=1.31)
    return {"name": request.param, "config": config, "torch": model,
            "path": path, "info": info}


# ------------------------------------------------------- 1. C vs reference


def test_runtime_matches_reference(exported):
    reference = ReferenceModel.load(exported["path"], activations=ACTIVATIONS)
    config = exported["config"]
    runtime = Model(exported["path"])
    session = runtime.session(max_options=8)

    for state, question, options in CASES:
        state = fit(state, config.max_state)
        question = fit(question, config.max_question)
        options = [fit(option, config.max_option) or b" " for option in options]
        if not state and not question:
            continue

        probabilities, result = session.decide(state, question, options)
        expected = reference.decide(state, question, options)

        assert np.allclose(probabilities, expected.probabilities,
                           atol=PROBABILITY_TOLERANCE), (
            f"{exported['name']} disagreed on {state[:32]!r}: "
            f"{probabilities} vs {expected.probabilities}")
        assert np.allclose(session.logits(), expected.logits, atol=LOGIT_TOLERANCE)
        assert result.argmax == int(np.argmax(expected.probabilities))

        # attention is over pooled positions, one distribution per option
        assert result.context_positions == expected.attention.shape[1]
        for index in range(len(options)):
            actual = np.array(session.attention(index))
            assert actual.shape == expected.attention[index].shape
            assert np.allclose(actual, expected.attention[index],
                               atol=10 * PROBABILITY_TOLERANCE if W8A8 else 1e-5)
            assert abs(actual.sum() - 1.0) < 1e-4

    session.close()
    runtime.close()


def test_runtime_geometry_matches_config(exported):
    runtime = Model(exported["path"])
    info = runtime.info
    config = exported["config"]
    assert info.name.decode() == config.name
    assert info.width == config.width
    assert info.blocks == config.blocks
    assert info.ffn_width == config.ffn_width
    assert info.rank == config.rank
    assert info.heads == config.heads
    assert info.stride == config.stride
    assert info.bigram_buckets == config.bigram_buckets
    assert info.parameters == config.parameter_count()
    assert abs(info.temperature - 1.31) < 1e-6
    runtime.close()


def test_cached_state_is_bit_identical(exported):
    """Reusing an encoded state must return exactly the same floats as
    re-encoding it. This is the property the split encoder exists for, and the
    question-to-state cross layer is one-directional precisely so that adding
    it does not break this: the state reads nothing, so its encoding and its
    cached keys and values stay a function of the state alone."""
    config = exported["config"]
    runtime = Model(exported["path"])
    session = runtime.session(max_options=8)

    state = fit(b"Order 4471 was refunded on 3 March but still shows a balance.",
                config.max_state)
    options = [fit(o, config.max_option) for o in (b"yes", b"no", b"unclear")]
    questions = [fit(q, config.max_question) for q in
                 (b"Was the refund issued?", b"Is the balance settled?", b"")]

    fresh = [session.decide(state, q, options)[0] for q in questions]

    session.set_state(state)
    session.set_menu(options)
    cached = [session.score(q)[0] for q in questions]

    assert fresh == cached
    # and the question must actually matter
    assert fresh[0] != fresh[1]
    session.close()
    runtime.close()


# --------------------------------------------------- 2. PyTorch vs reference


def test_torch_matches_reference_after_export(exported):
    """The exported container, read back, must reproduce the PyTorch graph.

    PyTorch runs with straight-through fake quantization enabled so that it
    evaluates the same rounded weights the container stores.
    """
    config = exported["config"]
    model: BlinkModel = exported["torch"]
    model.enable_fake_quant(True)
    model.eval()
    reference = ReferenceModel.load(exported["path"])
    collator = Collator(config, max_options=8)

    rows = []
    for state, question, options in CASES:
        state = state[:config.max_state] or b"s"
        question = question[:config.max_question] or b"q"
        options = [o[:config.max_option] or b" " for o in options]
        rows.append(Row(state.decode("utf-8", "replace"),
                        question.decode("utf-8", "replace"),
                        tuple(o.decode("utf-8", "replace") for o in options), 0))

    for row in rows:
        batch = collator([row])
        with torch.no_grad():
            logits = model(batch)[0, :len(row.options)].double().numpy()
        expected = reference.decide(
            row.state.encode()[:config.max_state],
            row.question.encode()[:config.max_question],
            [o.encode()[:config.max_option] for o in row.options],
        ).logits
        # Quantization of the *activations* is not performed anywhere, so the
        # two differ only by fp32 accumulation order inside PyTorch.
        assert np.allclose(logits, expected, atol=2e-3, rtol=2e-3), (
            f"{exported['name']}: torch {logits} vs reference {expected}")


def test_batched_torch_matches_single_row(exported):
    """Padding a batch must not change a row's logits: the runtime never pads."""
    config = exported["config"]
    model: BlinkModel = exported["torch"]
    model.eval()
    collator = Collator(config, max_options=8)

    rows = [
        Row("short state", "q?", ("a", "bb", "ccc"), 0),
        Row("a considerably longer state that forces padding of the first row",
            "a longer question as well", ("alpha", "b"), 1),
    ]
    with torch.no_grad():
        together = model(collator(rows))
        apart = [model(collator([row])) for row in rows]

    for index, row in enumerate(rows):
        width = len(row.options)
        assert np.allclose(together[index, :width].numpy(),
                           apart[index][0, :width].numpy(), atol=1e-4)


# ------------------------------------------------------- 3. hashing and I/O


def test_bigram_hash_agrees_everywhere(library):
    """All three implementations must index the same bucket, for every pair."""
    library.blink_bigram_index.argtypes = [ctypes.c_uint8, ctypes.c_uint8,
                                           ctypes.c_uint32]
    library.blink_bigram_index.restype = ctypes.c_uint32

    for buckets in (64, 256, 4096, 32768):
        table = bigram_table(buckets)
        assert table.shape == (256, 256)
        assert table.min() >= 0 and table.max() < buckets
        for previous in range(0, 256, 7):
            for current in range(0, 256, 5):
                pure = bigram_index(previous, current, buckets)
                assert pure == table[previous, current]
                assert pure == library.blink_bigram_index(previous, current,
                                                          buckets)


def test_bigram_hash_exhaustive_against_c(library):
    library.blink_bigram_index.argtypes = [ctypes.c_uint8, ctypes.c_uint8,
                                           ctypes.c_uint32]
    library.blink_bigram_index.restype = ctypes.c_uint32
    buckets = 4096
    table = bigram_table(buckets)
    for previous in range(256):
        for current in range(256):
            assert table[previous, current] == library.blink_bigram_index(
                previous, current, buckets)


def test_quantization_round_trip():
    rng = np.random.default_rng(7)
    matrix = rng.normal(0.0, 0.3, size=(37, 19))
    quantized, scale = quantize_rows(matrix)
    restored = dequantize_rows(quantized, scale)

    assert quantized.dtype == np.int8
    assert quantized.min() >= -127 and quantized.max() <= 127
    # every row must use the full int8 range, otherwise the scale is wrong
    assert np.abs(quantized).max(axis=1).min() == 127
    # the error is bounded by half a quantization step per element
    assert np.all(np.abs(restored - matrix) <= scale[:, None] / 2 + 1e-12)

    zeros = np.zeros((3, 5))
    quantized, scale = quantize_rows(zeros)
    assert np.all(quantized == 0)
    assert np.all(scale == 1.0)


def test_container_round_trip(exported):
    container = Container.load(exported["path"])
    assert container.config == exported["config"].__class__(
        **{**exported["config"].to_dict(), "temperature": container.config.temperature}
    )
    assert container.tensors["emb.unigram"].shape == (256, exported["config"].width)
    assert "stem.proj" in container.tensors
    assert "stem.proj.scale" in container.tensors

    data = bytearray(exported["path"].read_bytes())
    data[-1] ^= 0xFF
    with pytest.raises(ValueError, match="checksum"):
        Container(bytes(data))


@pytest.mark.parametrize("mixer_blocks,film,cross", [
    (0, False, False), (0, True, False), (1, False, False),
    (0, False, True), (1, True, True),
])
def test_optional_sublayers_can_be_absent(tmp_path, mixer_blocks, film, cross,
                                          library):
    """The self-attention sub-layer and the question conditioning are optional.

    Both live in header fields that were once reserved, so a container written
    before they existed reads them as zero and loads as the ablated model. That
    is what makes the format append-only, and it is also how `--no-film` and
    `--no-mixer` ship. The C runtime and the reference must agree in every
    combination, not only the one the presets use.
    """
    del library
    from dataclasses import replace

    torch.manual_seed(99)
    config = replace(preset("nano"), mixer_blocks=mixer_blocks, film=film,
                     cross=cross)
    model = BlinkModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)

    path = tmp_path / f"m{mixer_blocks}{int(film)}{int(cross)}.blink"
    info = export_model(model, path, temperature=1.0)
    assert info["parameters"] == config.parameter_count()

    runtime = Model(path)
    assert runtime.info.mixer_blocks == mixer_blocks
    assert bool(runtime.info.film) == film
    assert bool(runtime.info.cross) == cross

    reference = ReferenceModel.load(path, activations=ACTIVATIONS)
    session = runtime.session(max_options=4)
    state = fit(b"a state that both implementations must read alike",
                config.max_state)
    question = fit(b"which one?", config.max_question)
    options = [fit(o, config.max_option) for o in (b"first", b"second", b"third")]

    probabilities, _ = session.decide(state, question, options)
    expected = reference.decide(state, question, options)
    assert np.allclose(probabilities, expected.probabilities,
                       atol=PROBABILITY_TOLERANCE)

    # The question changes the answer in every configuration. Without FiLM it
    # does so only through the keys and values it contributes to the shared
    # context -- it can move where an option looks, but the model has no direct
    # way to make an option's query a function of it. Whether that indirect
    # path is enough is an empirical question, answered by the ablation in
    # docs/RESULTS.md, not by this test.
    other = fit(b"which other one entirely?", config.max_question)
    alternative, _ = session.decide(state, other, options)
    assert not np.allclose(probabilities, alternative, atol=1e-9)
    assert abs(sum(alternative) - 1.0) < 1e-5

    session.close()
    runtime.close()


def test_logits_are_bounded_whatever_the_weights(tmp_path, library):
    """The option head's logit is a sum of cosines, so it cannot exceed
    heads * |logit_scale| however large the projections grow.

    This is the property the cosine head exists for. Before it, one
    blink-small run grew its query, key and value projections twelvefold and
    reached a validation NLL of 160: nothing bounded the logit, so growing the
    weights kept paying. Here the projections are inflated a thousandfold and
    the bound must still hold.
    """
    del library
    torch.manual_seed(5)
    config = preset("nano")
    model = BlinkModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
        for projection in (model.wq, model.wk, model.wv):
            projection.weight.mul_(1000.0)
        model.logit_scale.fill_(2.5)

    path = tmp_path / "inflated.blink"
    export_model(model, path, temperature=1.0)
    runtime = Model(path)
    session = runtime.session(max_options=4)
    options = [fit(o, config.max_option) for o in (b"one", b"two", b"three")]
    session.decide(fit(b"any state at all", config.max_state),
                   fit(b"any question", config.max_question), options)
    bound = config.heads * 2.5
    for logit in session.logits():
        assert abs(logit) <= bound + 1e-5, (logit, bound)
    session.close()
    runtime.close()


def test_c_written_container_loads_in_python(tmp_path, library):
    """blink_synth writes containers from C; the Python reader must accept
    them, which keeps the two writers honest about the format."""
    del library
    tool = ROOT / "build" / "blink_synth"
    if not tool.exists():
        pytest.skip("run `make` first")
    for name in ("nano", "tiny"):
        path = tmp_path / f"{name}.blink"
        subprocess.run([str(tool), str(path), name], check=True,
                       capture_output=True)
        container = Container.load(path)
        assert container.config.name == f"blink-{name}"
        assert container.config.stride == preset(name).stride
        assert container.config.width == preset(name).width

        reference = ReferenceModel(container, activations=ACTIVATIONS)
        runtime = Model(path)
        session = runtime.session(max_options=4)
        state = b"synthetic weights still have to agree"[:container.config.max_state]
        options = [o[:container.config.max_option] for o in (b"one", b"two")]
        probabilities, _ = session.decide(state, b"", options)
        expected = reference.decide(state, b"", options)
        assert np.allclose(probabilities, expected.probabilities,
                           atol=PROBABILITY_TOLERANCE)
        session.close()
        runtime.close()


# ----------------------------------------------------------------- version


def test_versions_agree(library):
    """The C library, the Python package and pyproject.toml carry one
    version: BLINK_VERSION_STRING in include/blink.h is the source."""
    import re

    import blink_train

    header = (ROOT / "include" / "blink.h").read_text()
    numbers = [re.search(rf"#define BLINK_VERSION_{part} (\d+)", header).group(1)
               for part in ("MAJOR", "MINOR", "PATCH")]
    declared = ".".join(numbers)
    project = re.search(r'^version = "([^"]+)"',
                        (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert library.blink_version().decode() == declared
    assert blink_train.__version__ == declared
    assert project == declared
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert re.search(rf"^## \[?{re.escape(declared)}\]? ", changelog, re.M), (
        f"CHANGELOG.md has no section for {declared}")
