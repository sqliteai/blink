"""Train a Blink scorer and export it.

The objective is plain cross-entropy over the option set, which is a strictly
proper scoring rule: minimising it already pushes the probabilities towards
calibration rather than only towards the right argmax. After training, a single
temperature is fitted on the validation split by minimising validation NLL, and
that temperature is baked into the container. Calibration is then reported on
the untouched test split, never on the split it was fitted on.

With `--distill-alpha a` the training loss becomes

    (1 - a) * CE(logits, label) + a * CE(logits, target)

where `target` is a teacher's probability over the options, carried by every
training row (scripts/teacher_label.py). Everything downstream of the training
loss still uses the label: checkpoint selection on validation NLL, the
temperature fit and every metric. At a = 0 the loss is the plain one, term for
term.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from dataclasses import replace
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import BlinkConfig, preset
from .data import clip
from .data import Collator, Row, RowDataset, read_jsonl
from .export import export_model
from .model import BlinkModel


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean over rows of -sum_i t_i log p_i. Absent options carry t = 0 and a
    logit at the dtype's minimum; they are excluded rather than multiplied, so
    0 * -inf cannot turn the loss into NaN."""
    log_p = F.log_softmax(logits, dim=-1)
    terms = torch.where(targets > 0, targets * log_p, torch.zeros_like(log_p))
    return -terms.sum(dim=-1).mean()


def training_loss(logits: torch.Tensor, batch: dict, alpha: float) -> torch.Tensor:
    gold = F.cross_entropy(logits, batch["labels"])
    if alpha == 0.0:
        return gold
    soft = soft_cross_entropy(logits, batch["targets"])
    if alpha == 1.0:
        return soft
    return (1.0 - alpha) * gold + alpha * soft


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = move(batch, device)
        logits.append(model(batch).float().cpu())
        labels.append(batch["labels"].cpu())
    width = max(item.shape[1] for item in logits)
    padded = [
        F.pad(item, (0, width - item.shape[1]), value=torch.finfo(torch.float32).min)
        for item in logits
    ]
    return torch.cat(padded), torch.cat(labels)


TEMPERATURE_BOUNDS = (0.05, 20.0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, bool]:
    """Golden-section search over log-temperature, minimising NLL.

    Returns the temperature and whether it pinned against a search bound. A
    pinned fit is a failure signal, not a result: it means the logits are so
    badly scaled that the search ran out of room trying to undo them. One
    diverged run fitted 19.871 against a bound of 20.0 and the value was baked
    into the container without comment, which is how it went unnoticed.
    """
    def nll(log_t: float) -> float:
        scaled = logits / math.exp(log_t)
        return float(F.cross_entropy(scaled, labels))

    low, high = math.log(TEMPERATURE_BOUNDS[0]), math.log(TEMPERATURE_BOUNDS[1])
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = high - phi * (high - low), low + phi * (high - low)
    fa, fb = nll(a), nll(b)
    for _ in range(60):
        if fa < fb:
            high, b, fb = b, a, fa
            a = high - phi * (high - low)
            fa = nll(a)
        else:
            low, a, fa = a, b, fb
            b = low + phi * (high - low)
            fb = nll(b)
    temperature = math.exp((low + high) / 2.0)
    margin = 1.02
    pinned = (temperature < TEMPERATURE_BOUNDS[0] * margin
              or temperature > TEMPERATURE_BOUNDS[1] / margin)
    return temperature, pinned


def derive_limits(rows: list[Row], config: BlinkConfig,
                  ceiling: int = 2048) -> tuple[BlinkConfig, dict]:
    """Size the byte limits from the training corpus instead of guessing.

    The limits are a property of the trained container -- the position tables
    are indexed by byte offset -- so they cannot be raised at inference time.
    Guessing them wrong is not a small error: `blink-tiny`'s 256-byte state
    truncated every row of WANLI, whose premises reach 551 bytes.

    Each limit is the longest example in the training split, rounded up to a
    multiple of the pooling stride so the final window is full, and capped.
    """
    stride = config.stride

    def size(longest: int) -> int:
        rounded = ((max(longest, 1) + stride - 1) // stride) * stride
        return max(stride, min(rounded, ceiling))

    limits = {
        "max_state": size(max(len(r.state.encode()) for r in rows)),
        "max_question": size(max(len(r.question.encode()) for r in rows)),
        "max_option": size(max(len(o.encode()) for r in rows for o in r.options)),
    }
    return replace(config, **limits), limits


def coverage(rows: list[Row], config: BlinkConfig) -> dict:
    """What fraction of a split fits inside the limits without clipping."""
    fits = sum(
        1 for r in rows
        if len(r.state.encode()) <= config.max_state
        and len(r.question.encode()) <= config.max_question
        and all(len(o.encode()) <= config.max_option for o in r.options)
    )
    return {"rows": len(rows), "unclipped": fits, "fraction": fits / len(rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Blink scorer")
    parser.add_argument("train", help="training JSONL")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--no-film", action="store_true",
                        help="ablation: drop the question conditioning of the "
                             "option queries, leaving them question-independent")
    parser.add_argument("--no-mixer", action="store_true",
                        help="ablation: drop the self-attention sub-layers")
    parser.add_argument("--limits-from-data", action="store_true",
                        help="size max_state/max_question/max_option from the "
                             "training corpus rather than from the preset")
    parser.add_argument("--limit-ceiling", type=int, default=2048,
                        help="upper bound for --limits-from-data")
    parser.add_argument("--output", default="artifacts/blink-tiny.blink")
    parser.add_argument("--checkpoint", default=None,
                        help="where to write the fp32 weights; defaults to the "
                             "output path with a .pt suffix")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="do not write the fp32 weights. Only do this if "
                             "you are prepared to retrain: a change to the "
                             "container format then costs a full run rather "
                             "than a re-export")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--qat-fraction", type=float, default=0.3,
                        help="fraction of the final epochs trained with "
                             "straight-through int8 rounding")
    parser.add_argument("--max-options", type=int, default=16)
    parser.add_argument("--distill-alpha", type=float, default=0.0,
                        help="weight of the teacher-target cross-entropy in the "
                             "training loss, in [0, 1]; needs a `target` on "
                             "every training row")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    config: BlinkConfig = preset(args.preset)
    if args.no_film or args.no_mixer:
        config = replace(
            config,
            film=config.film and not args.no_film,
            mixer_blocks=0 if args.no_mixer else config.mixer_blocks,
        )

    train_rows = read_jsonl(args.train)
    validation_rows = read_jsonl(args.validation)
    if not 0.0 <= args.distill_alpha <= 1.0:
        raise SystemExit("--distill-alpha must lie in [0, 1]")
    if args.distill_alpha > 0.0:
        missing = sum(1 for row in train_rows if row.target is None)
        if missing:
            raise SystemExit(f"--distill-alpha needs a target on every training "
                             f"row; {missing} of {len(train_rows)} have none")

    derived = None
    if args.limits_from_data:
        config, derived = derive_limits(train_rows, config, args.limit_ceiling)

    device = select_device(args.device)
    collator = Collator(config, max_options=args.max_options)
    train_loader = DataLoader(RowDataset(train_rows), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator, drop_last=False)
    validation_loader = DataLoader(RowDataset(validation_rows),
                                   batch_size=args.batch_size, collate_fn=collator)

    model = BlinkModel(config).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    steps = max(1, args.epochs * len(train_loader))
    def learning_rate(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, learning_rate)
    quantize_from = int(args.epochs * (1.0 - args.qat_fraction))

    history = []
    best = {"nll": float("inf"), "state": None, "epoch": -1}
    started = time.perf_counter()

    for epoch in range(args.epochs):
        model.enable_fake_quant(epoch >= quantize_from)
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            batch = move(batch, device)
            loss = training_loss(model(batch), batch, args.distill_alpha)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach()) * batch["labels"].numel()
            count += batch["labels"].numel()

        logits, labels = collect_logits(model, validation_loader, device)
        validation_nll = float(F.cross_entropy(logits, labels))
        accuracy = float((logits.argmax(-1) == labels).float().mean())
        record = {
            "epoch": epoch + 1,
            "train_nll": total / count,
            "validation_nll": validation_nll,
            "validation_top1": accuracy,
            "fake_quant": epoch >= quantize_from,
        }
        history.append(record)
        if not args.quiet:
            print(json.dumps(record), flush=True)

        # Only checkpoints from the quantization-aware phase are comparable to
        # what gets exported, so prefer them once that phase has started.
        comparable = epoch >= quantize_from or quantize_from >= args.epochs
        if comparable and validation_nll < best["nll"]:
            best = {
                "nll": validation_nll,
                "epoch": epoch + 1,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.enable_fake_quant(True)
    logits, labels = collect_logits(model, validation_loader, device)
    temperature, temperature_pinned = fit_temperature(logits, labels)
    if temperature_pinned:
        print(json.dumps({
            "warning": "the calibration temperature pinned against its search "
                       "bound, which means the logits are badly scaled; treat "
                       "this checkpoint as failed rather than calibrated",
            "temperature": temperature,
            "bounds": list(TEMPERATURE_BOUNDS),
        }), flush=True)

    checkpoint_path = (None if args.no_checkpoint
                       else args.checkpoint or str(Path(args.output).with_suffix(".pt")))
    if checkpoint_path:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": config.to_dict(), "state_dict": model.state_dict(),
                    "temperature": temperature}, path)
    summary_checkpoint = checkpoint_path

    export = export_model(model, args.output, temperature=temperature)
    summary = {
        "preset": args.preset,
        "checkpoint": summary_checkpoint,
        "derived_limits": derived,
        "limits": {"max_state": config.max_state,
                   "max_question": config.max_question,
                   "max_option": config.max_option},
        "coverage": {"train": coverage(train_rows, config),
                     "validation": coverage(validation_rows, config)},
        "ablations": [name for name, on in
                      (("no-film", args.no_film), ("no-mixer", args.no_mixer))
                      if on],
        "config": config.to_dict(),
        "device": str(device),
        "seed": args.seed,
        "distill_alpha": args.distill_alpha,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "best_epoch": best["epoch"],
        "best_validation_nll": best["nll"],
        "temperature": temperature,
        "temperature_pinned": temperature_pinned,
        "train_seconds": time.perf_counter() - started,
        "export": export,
        "history": history,
    }
    if not args.quiet:
        print(json.dumps({k: v for k, v in summary.items() if k != "history"},
                         indent=2))
    return summary


if __name__ == "__main__":
    main()
