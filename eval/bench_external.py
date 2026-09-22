#!/usr/bin/env python3
"""Latency and memory of the external from-scratch baseline, measured the way
`bench/bench_latency.c` and `bench/bench_memory.c` measure Blink.

Same machine, one CPU thread, batch one, the same deterministic filler state
and the same menu of options. The baseline truncates its context at
`context_tokens` (256 bytes, question included), so longer states cost it
nothing more and are not measured. The baseline has no state cache, so every
decision is a full forward pass; it is compared with Blink's `decide` path and
with its `cached_state` path.

Memory is reported three ways: parameters as stored (float32), the checkpoint
on disk, and the peak resident set of the process, which for a PyTorch model
is dominated by the framework rather than by the weights.

    python eval/bench_external.py artifacts/external/external-w288-s7.pt \
        --json results/bench-external-w288.json
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

WORDS = ("order", "refund", "shipment", "invoice", "customer", "device",
         "network", "account", "ticket", "escalation", "warranty", "session")
OPTIONS = ("delivery and logistics", "billing and payments",
           "account access and sign-in", "device hardware repair",
           "privacy and data rights", "sales and product evaluation",
           "warranty replacement", "developer platform support",
           "enterprise onboarding", "fraud review",
           "returns processing", "network operations",
           "accessibility support", "legal and compliance",
           "partner integrations", "general enquiries")
QUESTION = "Which team should handle this ticket?"


def make_state(length: int) -> str:
    """The filler `bench_latency.c` generates, byte for byte."""
    out, index = [], 0
    while len(out) < length:
        for ch in WORDS[index % len(WORDS)]:
            if len(out) < length:
                out.append(ch)
        index += 1
        if len(out) < length:
            out.append(" ")
    return "".join(out)


def peak_rss_kib() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak // 1024 if sys.platform == "darwin" else peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--state-bytes", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--options", type=int, nargs="+", default=[4])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rss_start = peak_rss_kib()
    import torch

    torch.set_num_threads(1)
    from jevlike.data import ChoiceExample
    from jevlike.model import load_checkpoint
    rss_framework = peak_rss_kib()

    device = torch.device("cpu")
    model, collator, config = load_checkpoint(args.checkpoint, device)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    rows = []
    for state_bytes in args.state_bytes:
        state = make_state(state_bytes)
        for options in args.options:
            example = ChoiceExample(f"{QUESTION}\n{state}", OPTIONS[:options], 0)
            samples = []
            with torch.inference_mode():
                for i in range(args.warmup + args.samples):
                    started = time.perf_counter()
                    batch = collator([example])
                    model(batch).softmax(-1)[0, :options].tolist()
                    if i >= args.warmup:
                        samples.append(time.perf_counter() - started)
            samples.sort()
            row = {
                "model": f"external-w{config['width']}", "path": "decide",
                "state_bytes": state_bytes, "options": options,
                "context_tokens": config["context_tokens"],
                "samples": len(samples),
                "mean_us": round(statistics.fmean(samples) * 1e6, 3),
                "p50_us": round(samples[len(samples) // 2] * 1e6, 3),
                "p99_us": round(samples[int(len(samples) * 0.99)] * 1e6, 3),
                "decisions_per_second": round(1 / statistics.fmean(samples), 1),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    report = {
        "checkpoint": str(args.checkpoint),
        "config": config,
        "parameters": parameters,
        "weights_bytes_fp32": weight_bytes,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "rss_kib_interpreter": rss_start,
        "rss_kib_after_framework_import": rss_framework,
        "process_peak_rss_kib": peak_rss_kib(),
        "latency": rows,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "latency"}))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
