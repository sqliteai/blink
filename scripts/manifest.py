#!/usr/bin/env python3
"""Record what produced the results: checksums, versions and build flags.

A number is reproducible only if you can tell whether you reproduced it. This
writes the checksum of every input, artefact and report together with the
toolchain that made them, so two runs can be diffed rather than eyeballed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKED = ("artifacts", "data", "results", "build")
SKIP_SUFFIXES = (".o", ".a", ".log", ".pt")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def run(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=30, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def toolchain() -> dict:
    compiler = run("cc", "--version").splitlines()
    make = run("make", "-n", "--no-print-directory", "build/blink_kernels.o")
    flags = next((line for line in make.splitlines() if "-std=c99" in line), "")
    info = {
        "compiler": compiler[0] if compiler else "unknown",
        "compile_command": flags,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": run("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        "python": sys.version.split()[0],
        "git_revision": run("git", "rev-parse", "HEAD") or "not a git checkout",
        "git_dirty": bool(run("git", "status", "--porcelain")),
    }
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        info["torch"] = None
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    return info


def files() -> dict:
    out = {}
    for directory in TRACKED:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix in SKIP_SUFFIXES:
                continue
            if path.name.startswith("."):
                continue
            # Superseded runs and smoke runs are not evidence for anything
            # published; see .gitignore.
            if {"archive", "smoke"} & set(path.relative_to(base).parts[:-1]):
                continue
            relative = path.relative_to(ROOT).as_posix()
            out[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return out


def main() -> None:
    print(json.dumps({
        "toolchain": toolchain(),
        "files": files(),
    }, indent=2))


if __name__ == "__main__":
    main()
