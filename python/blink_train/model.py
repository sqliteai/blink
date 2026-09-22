"""PyTorch definition of the Blink scorer.

The forward pass here is the training-time twin of src/blink_runtime.c. Every
operation has a one-to-one counterpart in the runtime, including the zero
padding convention of the depthwise convolution and the masked mean used by the
global term. Padded positions are re-zeroed after every sub-layer so that a
padded batch computes exactly what the runtime computes on an unpadded
sequence.

Quantization-aware training is available through `enable_fake_quant`: the
tensors that ship as int8 are rounded through a straight-through estimator so
the exported model behaves like the trained one.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import SEG_OPTION, SEG_QUESTION, SEG_STATE, SEGMENTS, VOCAB, BlinkConfig
from .hashing import bigram_table


def quantize_dequantize(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric per-row int8 round trip, differentiable via straight-through."""
    flat = weight.reshape(weight.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    rounded = torch.clamp(torch.round(flat / scale), -127.0, 127.0) * scale
    out = rounded.reshape(weight.shape)
    return weight + (out - weight).detach()


class _FakeQuant(nn.Module):
    """Holds a tensor that ships as int8 and optionally rounds it in training."""

    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(tensor)
        self.fake_quant = False

    def forward(self) -> torch.Tensor:
        return quantize_dequantize(self.weight) if self.fake_quant else self.weight


def _unit(x: torch.Tensor) -> torch.Tensor:
    """Unit length along the last axis, leaving a zero vector at zero.

    Matches blink_l2_normalize, which returns a zero vector unchanged rather
    than dividing by zero.
    """
    norm = x.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, x / norm.clamp_min(1e-30), x)


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


def _layer_norm(x: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), gain, bias, eps=1e-5)


class Stem(nn.Module):
    """Full-resolution byte mixing, run once before the pooling step.

    A depthwise convolution supplies local n-grams and a pointwise projection
    supplies channel mixing. There is no feed-forward here on purpose: the
    expensive part belongs behind the pool.
    """

    def __init__(self, config: BlinkConfig) -> None:
        super().__init__()
        w, k = config.width, config.conv_width
        self.kernel = k
        self.ln_gain = nn.Parameter(torch.ones(w))
        self.ln_bias = nn.Parameter(torch.zeros(w))
        self.conv = nn.Parameter(torch.zeros(k, w))
        self.proj = _FakeQuant(torch.zeros(w, w))
        with torch.no_grad():
            self.conv.normal_(0.0, 1.0 / k**0.5)
            nn.init.normal_(self.proj.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = _layer_norm(x, self.ln_gain, self.ln_bias) * mask
        local = F.conv1d(
            a.transpose(1, 2),
            self.conv.t().reshape(x.shape[-1], 1, self.kernel),
            groups=x.shape[-1],
            padding=self.kernel // 2,
        ).transpose(1, 2)
        return (x + local + a @ self.proj().t()) * mask


def pool(x: torch.Tensor, mask: torch.Tensor, stride: int):
    """Mean-pool non-overlapping windows of `stride` positions.

    A window is divided by the number of real bytes it contains, matching
    blink_pool in the runtime, where sequences carry no padding at all and the
    final window is simply short.
    """
    if stride <= 1:
        return x, mask
    batch, length, width = x.shape
    padding = (-length) % stride
    if padding:
        x = F.pad(x, (0, 0, 0, padding))
        mask = F.pad(mask, (0, 0, 0, padding))
    positions = x.shape[1] // stride
    windows = x.reshape(batch, positions, stride, width)
    counts = mask.reshape(batch, positions, stride, 1).sum(dim=2)
    pooled = windows.sum(dim=2) / counts.clamp_min(1.0)
    return pooled, (counts > 0).float()


class Mixer(nn.Module):
    """Self-attention over the pooled positions.

    Everything else in the encoder is linear in the sequence length: a
    depthwise convolution reaches a few neighbours and a masked mean reaches
    everything without distinguishing anything. This is the one path that can
    bind two specific distant positions, which is what tasks like "which city
    does this person live in" need. It is affordable only because it runs after
    the pool, where the sequence is a fraction of its original length.
    """

    def __init__(self, config: BlinkConfig) -> None:
        super().__init__()
        w = config.width
        self.heads = config.heads
        self.head_dim = w // config.heads
        self.ln_gain = nn.Parameter(torch.ones(w))
        self.ln_bias = nn.Parameter(torch.zeros(w))
        self.q = _FakeQuant(torch.empty(w, w))
        self.k = _FakeQuant(torch.empty(w, w))
        self.v = _FakeQuant(torch.empty(w, w))
        self.o = _FakeQuant(torch.empty(w, w))
        with torch.no_grad():
            for projection in (self.q, self.k, self.v):
                nn.init.normal_(projection.weight, 0.0, w**-0.5)
            # The output projection is zero, so the whole sub-layer starts as
            # an exact identity on the residual stream. An attention layer
            # added with a random output projection perturbs a working path
            # from the first step and has to climb back out; that cost this
            # model roughly thirty epochs in a worse optimum than it started
            # in, twice, before the pattern was recognised. Same reasoning as
            # the zero-initialised FiLM weights.
            nn.init.zeros_(self.o.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        a = _layer_norm(x, self.ln_gain, self.ln_bias) * mask

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, length, self.heads,
                                  self.head_dim).transpose(1, 2)

        q = split(a @ self.q().t())
        k = split(a @ self.k().t())
        v = split(a @ self.v().t())
        scores = (q @ k.transpose(-1, -2)) * self.head_dim**-0.5
        keep = mask.squeeze(-1) > 0
        scores = scores.masked_fill(~keep[:, None, None, :],
                                    torch.finfo(scores.dtype).min)
        attended = (scores.softmax(dim=-1) @ v)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return (x + attended @ self.o().t()) * mask


class Cross(nn.Module):
    """One-directional attention from the question to the state.

    The state and the question are separate sequences, so nothing in the
    encoder otherwise compares a question byte against a state byte -- they
    meet only in the option head, which takes a weighted average over both, and
    an average cannot express "are these the same word". Deciding whether *"the
    order was cancelled"* supports or contradicts *"the order was cancelled on
    Tuesday"* is exactly that comparison.

    The question reads the state; the state never reads the question. So the
    state encoding remains a function of the state alone and stays exactly
    cacheable.
    """

    def __init__(self, config: BlinkConfig) -> None:
        super().__init__()
        w = config.width
        self.heads = config.heads
        self.head_dim = w // config.heads
        self.ln_gain = nn.Parameter(torch.ones(w))
        self.ln_bias = nn.Parameter(torch.zeros(w))
        self.q = _FakeQuant(torch.empty(w, w))
        self.k = _FakeQuant(torch.empty(w, w))
        self.v = _FakeQuant(torch.empty(w, w))
        self.o = _FakeQuant(torch.empty(w, w))
        with torch.no_grad():
            for projection in (self.q, self.k, self.v):
                nn.init.normal_(projection.weight, 0.0, w**-0.5)
            # The output projection is zero, so the whole sub-layer starts as
            # an exact identity on the residual stream. An attention layer
            # added with a random output projection perturbs a working path
            # from the first step and has to climb back out; that cost this
            # model roughly thirty epochs in a worse optimum than it started
            # in, twice, before the pattern was recognised. Same reasoning as
            # the zero-initialised FiLM weights.
            nn.init.zeros_(self.o.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                state: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        a = _layer_norm(x, self.ln_gain, self.ln_bias) * mask
        b = _layer_norm(state, self.ln_gain, self.ln_bias) * state_mask

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(tensor.shape[0], tensor.shape[1], self.heads,
                                  self.head_dim).transpose(1, 2)

        q = split(a @ self.q().t())
        k = split(b @ self.k().t())
        v = split(b @ self.v().t())
        scores = (q @ k.transpose(-1, -2)) * self.head_dim**-0.5
        keep = state_mask.squeeze(-1) > 0
        # A row whose state is entirely padding has nothing to attend to; keep
        # the softmax finite and mask its contribution away afterwards.
        empty = ~keep.any(dim=1, keepdim=True)
        scores = scores.masked_fill(~(keep | empty)[:, None, None, :],
                                    torch.finfo(scores.dtype).min)
        attended = (scores.softmax(dim=-1) @ v)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        attended = attended * (~empty).float().unsqueeze(-1)
        return (x + attended @ self.o().t()) * mask


class Block(nn.Module):
    """Local depthwise mixing plus one global mean term, then a feed-forward.

    The trailing `mixer_blocks` blocks of the model additionally carry a
    self-attention sub-layer; see `Mixer`.
    """

    def __init__(self, config: BlinkConfig, mixer: bool = False) -> None:
        super().__init__()
        w, f, k = config.width, config.ffn_width, config.conv_width
        self.kernel = k
        self.ln1_gain = nn.Parameter(torch.ones(w))
        self.ln1_bias = nn.Parameter(torch.zeros(w))
        self.conv = nn.Parameter(torch.zeros(k, w))
        with torch.no_grad():
            self.conv.normal_(0.0, 1.0 / k**0.5)
        self.glob = _FakeQuant(torch.zeros(w, w))
        self.ln2_gain = nn.Parameter(torch.ones(w))
        self.ln2_bias = nn.Parameter(torch.zeros(w))
        self.fc1 = _FakeQuant(torch.empty(f, w))
        self.fc2 = _FakeQuant(torch.empty(w, f))
        with torch.no_grad():
            nn.init.normal_(self.glob.weight, 0.0, 0.02)
            nn.init.normal_(self.fc1.weight, 0.0, w**-0.5)
            nn.init.normal_(self.fc2.weight, 0.0, f**-0.5)
        self.mixer = Mixer(config) if mixer else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: [batch, length, 1], 1.0 on real bytes.
        a = _layer_norm(x, self.ln1_gain, self.ln1_bias) * mask

        counts = mask.sum(dim=1).clamp_min(1.0)
        global_mean = a.sum(dim=1) / counts
        global_term = global_mean @ self.glob().t()

        local = F.conv1d(
            a.transpose(1, 2),
            self.conv.t().reshape(x.shape[-1], 1, self.kernel),
            groups=x.shape[-1],
            padding=self.kernel // 2,
        ).transpose(1, 2)

        x = (x + local + global_term.unsqueeze(1)) * mask
        b = _layer_norm(x, self.ln2_gain, self.ln2_bias) * mask
        x = (x + _relu2(b @ self.fc1().t()) @ self.fc2().t()) * mask
        if self.mixer is not None:
            x = self.mixer(x, mask)
        return x


class BlinkModel(nn.Module):
    def __init__(self, config: BlinkConfig) -> None:
        super().__init__()
        self.config = config
        w = config.width
        self.unigram = _FakeQuant(torch.zeros(VOCAB, w))
        self.bigram = _FakeQuant(torch.zeros(config.bigram_buckets, w))
        self.pos_state = _FakeQuant(torch.zeros(config.max_state, w))
        self.pos_question = _FakeQuant(torch.zeros(config.max_question, w))
        self.pos_option = _FakeQuant(torch.zeros(config.max_option, w))
        self.segment = nn.Parameter(torch.zeros(SEGMENTS, w))
        with torch.no_grad():
            for table in (self.unigram, self.bigram, self.pos_state,
                          self.pos_question, self.pos_option):
                nn.init.normal_(table.weight, 0.0, 0.05)

        self.stem = Stem(config)
        self.cross = Cross(config) if config.cross else None
        self.blocks = nn.ModuleList(
            Block(config, mixer=index >= config.blocks - config.mixer_blocks)
            for index in range(config.blocks)
        )

        self.ctx_ln_gain = nn.Parameter(torch.ones(w))
        self.ctx_ln_bias = nn.Parameter(torch.zeros(w))
        self.opt_ln_gain = nn.Parameter(torch.ones(w))
        self.opt_ln_bias = nn.Parameter(torch.zeros(w))
        # Feature-wise linear modulation of the option vectors by the
        # question. See the note in src/blink_internal.h: without this the
        # score has no question-by-option interaction term at all.
        if config.film:
            self.film_ln_gain = nn.Parameter(torch.ones(w))
            self.film_ln_bias = nn.Parameter(torch.zeros(w))
            # Zero-initialised on purpose: gamma = beta = 0 makes the
            # modulation an exact identity, so training starts from the
            # unconditioned model and can only move away from it deliberately.
            # Random initialisation perturbs a working residual path from step
            # one, and in a 100-epoch run that cost roughly twenty epochs in a
            # bad optimum before it recovered.
            self.film_gamma = nn.Parameter(torch.zeros(w, w))
            self.film_beta = nn.Parameter(torch.zeros(w, w))
        self.wq = _FakeQuant(torch.empty(config.rank, w))
        self.wk = _FakeQuant(torch.empty(config.rank, w))
        self.wv = _FakeQuant(torch.empty(config.rank, w))
        with torch.no_grad():
            for projection in (self.wq, self.wk, self.wv):
                nn.init.normal_(projection.weight, 0.0, w**-0.5)
        self.logit_scale = nn.Parameter(torch.ones(1))

        self.register_buffer(
            "bigram_lookup",
            torch.from_numpy(bigram_table(config.bigram_buckets)),
            persistent=False,
        )

    # ------------------------------------------------------------ helpers

    def enable_fake_quant(self, enabled: bool = True) -> None:
        for module in self.modules():
            if isinstance(module, _FakeQuant):
                module.fake_quant = enabled

    def quantized_modules(self) -> dict[str, _FakeQuant]:
        return {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, _FakeQuant)
        }

    def _embed(self, ids: torch.Tensor, mask: torch.Tensor, segment: int,
               positions: _FakeQuant) -> torch.Tensor:
        length = ids.shape[1]
        previous = torch.zeros_like(ids)
        previous[:, 1:] = ids[:, :-1]
        buckets = self.bigram_lookup[previous.long(), ids.long()]
        x = (
            F.embedding(ids.long(), self.unigram())
            + F.embedding(buckets, self.bigram())
            + positions()[:length].unsqueeze(0)
            + self.segment[segment]
        )
        return x * mask

    def encode(self, ids: torch.Tensor, mask: torch.Tensor, segment: int,
               positions: _FakeQuant, state=None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the pooled hidden states and their pooled mask.

        `state` is the encoded state, supplied only when encoding the question
        so it can attend to it; see `Cross`.
        """
        x = self._embed(ids, mask, segment, positions)
        x = self.stem(x, mask)
        x, mask = pool(x, mask, self.config.stride)
        for block in self.blocks:
            x = block(x, mask)
        if self.cross is not None and state is not None:
            x = self.cross(x, mask, state[0], state[1])
        return x, mask

    # ------------------------------------------------------------- forward

    def context_kv(self, batch: dict):
        """Keys, values, the combined mask, and the question summary."""
        state, state_mask = self.encode(
            batch["state_ids"], batch["state_mask"].unsqueeze(-1).float(),
            SEG_STATE, self.pos_state,
        )
        question, question_mask = self.encode(
            batch["question_ids"], batch["question_mask"].unsqueeze(-1).float(),
            SEG_QUESTION, self.pos_question, state=(state, state_mask),
        )
        hidden = torch.cat([state, question], dim=1)
        mask = torch.cat([state_mask, question_mask], dim=1).squeeze(-1) > 0
        summary = ((question * question_mask).sum(dim=1)
                   / question_mask.sum(dim=1).clamp_min(1.0))
        present = question_mask.sum(dim=1) > 0
        normed = _layer_norm(hidden, self.ctx_ln_gain, self.ctx_ln_bias)
        return (normed @ self.wk().t(), normed @ self.wv().t(), mask,
                summary * present)

    def option_vectors(self, batch: dict) -> torch.Tensor:
        """Pooled per-option encodings, before any question conditioning."""
        ids = batch["option_ids"]
        mask = batch["option_mask"].float()
        b, n, length = ids.shape
        flat_ids = ids.reshape(b * n, length)
        flat_mask = mask.reshape(b * n, length).unsqueeze(-1)
        encoded, encoded_mask = self.encode(flat_ids, flat_mask, SEG_OPTION,
                                            self.pos_option)
        pooled = (encoded * encoded_mask).sum(dim=1) / encoded_mask.sum(dim=1).clamp_min(1.0)
        return pooled.reshape(b, n, -1)

    def option_queries(self, vectors: torch.Tensor,
                       summary: torch.Tensor) -> torch.Tensor:
        if self.config.film:
            normed = _layer_norm(summary, self.film_ln_gain, self.film_ln_bias)
            gamma = (normed @ self.film_gamma.t()).unsqueeze(1)
            beta = (normed @ self.film_beta.t()).unsqueeze(1)
            # An empty question gives a zero summary, hence gamma = beta = 0
            # and the option vectors pass through untouched.
            vectors = vectors * (1.0 + gamma) + beta
        pooled = _layer_norm(vectors, self.opt_ln_gain, self.opt_ln_bias)
        return pooled @ self.wq().t()

    def forward(self, batch: dict, shuffle_context: bool = False) -> torch.Tensor:
        config = self.config
        keys, values, context_mask, summary = self.context_kv(batch)
        if shuffle_context and keys.shape[0] > 1:
            keys = keys.roll(1, dims=0)
            values = values.roll(1, dims=0)
            context_mask = context_mask.roll(1, dims=0)
        queries = self.option_queries(self.option_vectors(batch), summary)

        b, n, rank = queries.shape
        length = keys.shape[1]
        heads, head_dim = config.heads, config.head_dim
        q = queries.reshape(b, n, heads, head_dim).transpose(1, 2)
        k = keys.reshape(b, length, heads, head_dim).transpose(1, 2)
        v = values.reshape(b, length, heads, head_dim).transpose(1, 2)

        # Cosine head: queries, keys and the attended vector are unit length,
        # so the scores and the logit are bounded however large the
        # projections grow. Without this a run can grow |q|, |k| and |v| to
        # sharpen attention and raise confidence, and nothing stops it; one
        # blink-small run reached a validation NLL of 160 that way. See
        # blink_l2_normalize in src/blink_kernels.c.
        span = head_dim**0.5
        q = _unit(q)
        k = _unit(k)
        scores = torch.einsum("bhnd,bhld->bhnl", q, k) * span
        scores = scores.masked_fill(
            ~context_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = scores.softmax(dim=-1)
        attended = _unit(torch.einsum("bhnl,bhld->bhnd", attention, v))
        logits = (q * attended).sum(-1)
        logits = logits.sum(dim=1) * self.logit_scale

        return logits.masked_fill(
            ~batch["option_present"], torch.finfo(logits.dtype).min
        )
