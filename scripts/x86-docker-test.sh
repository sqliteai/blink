#!/usr/bin/env bash
# Build, test and time the x86-64 SIMD paths in a linux/amd64 container.
#
# Meant for machines that are not x86-64 themselves (Apple silicon, where
# Docker Desktop runs amd64 containers through Rosetta), and equally usable on
# a real x86 host, where the timings become meaningful. Three stages:
#
#   tests    the C suite for the default build, W8A8, SIMD compiled out, -O0
#            and AddressSanitizer + UBSan; test_kernels forces every x86 level
#            (scalar, SSE2, AVX2) in turn inside each build
#   parity   a trained model on test rows, at every level, against the float64
#            reference computed on the host (needs .venv with numpy and the
#            corpus; skipped otherwise), and W8A8 SSE2 against W8A8 AVX2
#   bench    bench_latency at every level, W8A32 and W8A8
#
# The repository is mounted read-only and copied inside the container, so
# nothing in build/ on the host is touched.
#
#   bash scripts/x86-docker-test.sh                    # all three stages
#   bash scripts/x86-docker-test.sh --skip-bench
#   bash scripts/x86-docker-test.sh --image gcc:14 --rows 300
#
# The image is pulled on first use (gcc:14 for linux/amd64 is about 1.2 GB).
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=gcc:14
MODEL=artifacts/blink-tiny-synthetic-s7.blink
DATA=data/synthetic/test.jsonl
ROWS=300
PYTHON=.venv/bin/python
RUN_TESTS=1 RUN_PARITY=1 RUN_BENCH=1

while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --data) DATA=$2; shift 2 ;;
    --rows) ROWS=$2; shift 2 ;;
    --skip-tests) RUN_TESTS=0; shift ;;
    --skip-parity) RUN_PARITY=0; shift ;;
    --skip-bench) RUN_BENCH=0; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/blink-x86.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

if [ "$RUN_PARITY" = 1 ]; then
  if [ ! -x "$PYTHON" ] || [ ! -f "$MODEL" ] || [ ! -f "$DATA" ]; then
    echo "parity: needs $PYTHON, $MODEL and $DATA; skipping it"
    RUN_PARITY=0
  fi
fi

# ------------------------------------------------------------ host: parity inputs

if [ "$RUN_PARITY" = 1 ]; then
  cp "$MODEL" "$WORK/model.blink"
  # Rows as length-prefixed bytes, plus the float64 reference for both
  # activation modes, computed here where numpy is available.
  PYTHONPATH=python "$PYTHON" - "$WORK" "$MODEL" "$DATA" "$ROWS" <<'EOF'
import struct, sys
import numpy as np
from blink_train.data import read_jsonl
from blink_train.reference import ReferenceModel

work, model, data, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
fp32 = ReferenceModel.load(model)
int8 = ReferenceModel.load(model, activations="int8")
config = fp32.config
references = {"fp32": [], "int8": []}
with open(f"{work}/rows.bin", "wb") as out:
    for row in read_jsonl(data)[:count]:
        state = row.state.encode()[:config.max_state]
        question = row.question.encode()[:config.max_question]
        options = [o.encode()[:config.max_option] for o in row.options]
        out.write(struct.pack("<I", len(state)) + state)
        out.write(struct.pack("<I", len(question)) + question)
        out.write(struct.pack("<I", len(options)))
        for option in options:
            out.write(struct.pack("<I", len(option)) + option)
        references["fp32"].append(fp32.decide(state, question, options).probabilities)
        references["int8"].append(int8.decide(state, question, options).probabilities)
for name, rows in references.items():
    with open(f"{work}/reference-{name}.txt", "w") as out:
        for probabilities in rows:
            out.write(" ".join(f"{p:.12g}" for p in probabilities) + "\n")
print(f"parity: {len(references['fp32'])} rows from {data}")
EOF
fi

# Reads rows.bin and prints one line of probabilities per row.
cat >"$WORK/decide_rows.c" <<'EOF'
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "blink.h"

static FILE *in;
static uint32_t u32(void)
{
    uint32_t v;
    if (fread(&v, 4, 1, in) != 1) exit(0);
    return v;
}
static char *take(uint32_t n)
{
    char *b = malloc(n + 1u);
    if (!b || (n && fread(b, 1, n, in) != n)) exit(3);
    b[n] = 0;
    return b;
}
int main(int argc, char **argv)
{
    blink_status status;
    blink_model *model = argc > 2 ? blink_model_open_file(argv[1], 1, &status) : NULL;
    if (!model) return 1;
    blink_limits limits = {0, 0, 64, 0};
    blink_session *session = blink_session_create(model, &limits, &status);
    in = fopen(argv[2], "rb");
    if (!session || !in) return 1;
    for (;;) {
        uint32_t sl = u32(); char *state = take(sl);
        uint32_t ql = u32(); char *question = take(ql);
        uint32_t n = u32();
        const char *options[64]; size_t lengths[64]; float p[64];
        for (uint32_t i = 0; i < n; ++i) { lengths[i] = u32(); options[i] = take((uint32_t)lengths[i]); }
        if (blink_decide(session, state, sl, question, ql, options, lengths, n, p, NULL) != BLINK_OK) return 4;
        for (uint32_t i = 0; i < n; ++i) printf("%.9g%c", p[i], i + 1 < n ? ' ' : '\n');
        free(state); free(question);
        for (uint32_t i = 0; i < n; ++i) free((void *)options[i]);
    }
}
EOF

# ------------------------------------------------------------ container

cat >"$WORK/inside.sh" <<'EOF'
set -uo pipefail
mkdir -p /work && cd /work
cp -r /src/src /src/include /src/tests /src/tools /src/bench /src/examples /src/Makefile .
echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
echo "features: $(grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -E '^(sse2|ssse3|sse4_1|avx|avx2|fma|avx512f|avx512_vnni|avx_vnni)$' | tr '\n' ' ')"
case "$(grep -m1 'model name' /proc/cpuinfo)" in
  *VirtualApple*) echo "note: emulated by Rosetta; timings are not x86 timings" ;;
esac
status=0

if [ "$RUN_TESTS" = 1 ]; then
  suite() {
    name=$1; shift
    rm -rf build build-w8a8
    if make -j8 "$@" test >/tmp/log 2>&1; then result=ok; else result=FAIL; status=1; fi
    levels=$(grep -h -E 'x86 levels tested' build*/test_kernels.log 2>/dev/null | head -1)
    printf '  %-22s %-4s %s\n' "$name" "$result" "$levels"
    [ "$result" = ok ] || grep -E 'FAIL|error:|test_.*[1-9][0-9]* failures' /tmp/log | head -20
  }
  echo "== tests"
  suite "default"
  suite "W8A8" W8A8=1
  suite "scalar only" OPT='-O3 -DBLINK_SCALAR_ONLY=1'
  suite "W8A8 scalar only" W8A8=1 OPT='-O3 -DBLINK_SCALAR_ONLY=1'
  suite "-O0" OPT=-O0
  suite "ASan + UBSan" OPT='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
        LDLIBS='-lm -fsanitize=address,undefined'
  suite "W8A8 ASan + UBSan" W8A8=1 OPT='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
        LDLIBS='-lm -fsanitize=address,undefined'
fi

rm -rf build build-w8a8
make -j8 build/libblink.a >/dev/null 2>&1 && make -j8 W8A8=1 build-w8a8/libblink.a >/dev/null 2>&1 || status=1

if [ "$RUN_PARITY" = 1 ]; then
  cc -std=c99 -O2 -Iinclude /w/decide_rows.c build/libblink.a -lm -o decide
  cc -std=c99 -O2 -Iinclude /w/decide_rows.c build-w8a8/libblink.a -lm -o decide8
  for level in scalar sse2 avx2; do
    BLINK_X86_SIMD=$level ./decide /w/model.blink /w/rows.bin >/w/x86-$level.txt || status=1
  done
  for kernel in sse2 avx2; do
    BLINK_W8A8_KERNEL=$kernel ./decide8 /w/model.blink /w/rows.bin >/w/x86-w8a8-$kernel.txt || status=1
  done
fi

if [ "$RUN_BENCH" = 1 ]; then
  make -j8 >/dev/null 2>&1 && make fixtures >/dev/null && make -j8 W8A8=1 >/dev/null 2>&1 && make W8A8=1 fixtures >/dev/null
  for level in scalar sse2 avx2; do
    BLINK_X86_SIMD=$level ./build/bench_latency 2>/dev/null | sed "s/^/$level /"
  done >/w/bench.txt
  for kernel in scalar sse2 avx2; do
    BLINK_W8A8_KERNEL=$kernel ./build-w8a8/bench_latency 2>/dev/null | sed "s/^/w8a8-$kernel /"
  done >>/w/bench.txt
fi
exit $status
EOF

container_status=0
docker run --rm --platform linux/amd64 \
  -e RUN_TESTS="$RUN_TESTS" -e RUN_PARITY="$RUN_PARITY" -e RUN_BENCH="$RUN_BENCH" \
  -v "$PWD":/src:ro -v "$WORK":/w "$IMAGE" bash /w/inside.sh || container_status=$?

# ------------------------------------------------------------ host: report

if [ "$RUN_PARITY" = 1 ]; then
  echo "== parity"
  "$PYTHON" - "$WORK" <<'EOF' || container_status=1
import sys
import numpy as np

work = sys.argv[1]
def load(name):
    with open(f"{work}/{name}.txt") as f:
        return [np.array([float(v) for v in line.split()]) for line in f]

ok = True
def compare(name, reference, tolerance):
    global ok
    got, ref = load(name), load(reference)
    worst = max(np.abs(a - b).max() for a, b in zip(got, ref))
    flips = sum(int(np.argmax(a) != np.argmax(b)) for a, b in zip(got, ref))
    passed = len(got) == len(ref) and worst <= tolerance
    ok &= passed
    print(f"  {name:14s} vs {reference:16s} rows {len(got):4d}  max|dp| {worst:.2e}"
          f"  argmax changed {flips}  {'ok' if passed else 'FAIL'}")

for level in ("scalar", "sse2", "avx2"):
    compare(f"x86-{level}", "reference-fp32", 1e-5)
for kernel in ("sse2", "avx2"):
    compare(f"x86-w8a8-{kernel}", "reference-int8", 1e-3)
same = open(f"{work}/x86-w8a8-sse2.txt").read() == open(f"{work}/x86-w8a8-avx2.txt").read()
print(f"  W8A8 SSE2 and AVX2 byte-identical: {'yes' if same else 'NO'}")
sys.exit(0 if ok and same else 1)
EOF
fi

if [ "$RUN_BENCH" = 1 ] && [ -s "$WORK/bench.txt" ]; then
  echo "== bench (p50, 4 options)"
  python3 - "$WORK/bench.txt" <<'EOF'
import json, sys
rows = {}
for line in open(sys.argv[1]):
    tag, payload = line.split(" ", 1)
    r = json.loads(payload)
    rows[(tag, r["model"], r["path"], r["state_bytes"], r["options"])] = r["p50_us"]
tags = ["scalar", "sse2", "avx2", "w8a8-scalar", "w8a8-sse2", "w8a8-avx2"]
print("  " + " " * 30 + "".join(f"{t:>13s}" for t in tags))
for model, state in (("blink-tiny", 256), ("blink-small", 512)):
    for path in ("decide", "cached_state", "new_menu"):
        cells = [rows.get((t, model, path, state, 4)) for t in tags]
        print(f"  {model + ' ' + path + ' ' + str(state):30s}"
              + "".join(f"{c:>10.0f} us" if c is not None else f"{'-':>13s}" for c in cells))
EOF
fi

exit "$container_status"
