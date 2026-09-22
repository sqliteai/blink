# Changelog

All notable changes to Blink are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). The version is
defined once, in `include/blink.h` (`BLINK_VERSION_MAJOR`, `_MINOR`, `_PATCH`);
`blink_version()`, `blink --version`, `blink_train.__version__` and
`pyproject.toml` all report it, and a test fails if they disagree.

Two other numbers move independently of the release: `BLINK_ABI_VERSION` (1),
bumped when the C interface changes incompatibly and carried in the shared
library's name, and the container format version (3), bumped when a `.blink`
file written by one release can no longer be read by another.

## [0.1.0] - 2026-09-22

First release.

### Added

**Runtime**

- A one-pass typed-decision model: given a state, a question and a list of
  options declared at call time, one forward pass returns one probability per
  option. No token generation, no decoding loop, nothing to parse.
- A C99 runtime with no dependencies beyond libc and libm. Weights are mapped
  read-only and never copied, a session lives in one arena whose size is known
  in advance (`blink_session_size`), and scoring allocates nothing:
  `tests/c/check_no_malloc.sh` fails if the scoring objects reference an
  allocator.
- The C API in `include/blink.h`: `blink_state_set` and `blink_menu_set` encode
  once, `blink_score` and `blink_score_batch` are the cheap per-question path,
  `blink_decide` does all three. A cached state is reused exactly, bit for bit.
  Up to 64 options per decision.
- The `.blink` container, format 3: int8 weights with one fp32 scale per row,
  a CRC-32, and a loader that refuses malformed or truncated files with a
  specific error.
- The architecture: byte-level input with hashed bigrams, a full-resolution
  stem, pooling, convolution and feed-forward blocks with a self-attention
  mixer, question-to-state cross attention, FiLM conditioning of the option
  queries on the question, and a multi-head cosine option head whose logits
  are bounded by construction. The reasons are in `docs/ARCHITECTURE.md`.
- Two presets: `blink-tiny` (407k parameters, 452 KiB), the recommended one,
  and `blink-small` (7.9M), experimental and not shipped.

**Numeric backends**

- NEON on Arm and, on x86-64, SSE2 or AVX2 + FMA chosen at run time
  (`BLINK_X86_SIMD` forces one). The scalar loop is the normative definition;
  every SIMD path agrees with a float64 reference to within 1e-5.
- `make W8A8=1`: int8 activations as well as weights, with SDOT, I8MM, SSE2 or
  AVX2 integer kernels that are bit-identical to one another
  (`BLINK_W8A8_KERNEL` forces one). About 1.8× faster on blink-tiny and 3×
  on blink-small, with no measurable change in accuracy on any published
  model, and still no allocation while scoring.
- `make ACCELERATE=1` on macOS: projections, layer norms and attention through
  Apple's Accelerate framework. The fastest fresh decision measured
  (172 µs for blink-tiny), at the cost of allocations inside Accelerate and a
  dense fp32 copy of the projections.
- `blink_backend()` reports which of these is running.

**WebAssembly**

- `make wasm`: the same sources as a 66 KB `wasm32-wasip1` reactor with seven
  stubbed WASI imports and no file system, and `wasm/blink.mjs` to load it in
  browsers and Node. `make test-wasm` checks it against the native build.

**Command line**

- `blink`: one decision from the shell as one line of JSON, `--info` for a
  model's geometry, `--version` for the library version and backend, `--help`.
  Documented in `docs/CLI.md` and the `blink(1)` man page.

**Installation**

- `make install` and `make uninstall`, with `PREFIX` and `DESTDIR`: the `blink`
  tool, `libblink.a`, the shared library (`libblink.so.1`,
  `libblink.1.dylib`), `blink.h`, `blink.pc` for pkg-config and the man page.

**Training and evaluation**

- `python/blink_train`: a PyTorch trainer with quantization-aware training for
  the weights, temperature calibration on validation, the exporter, a float64
  reference implementation of the forward pass, and a ctypes binding to the C
  runtime (`BLINK_LIBRARY` selects a build).
- `eval/evaluate.py`: accuracy, balanced accuracy, calibration, shuffled-state
  and shuffled-question controls, perturbations, per-family strata and group
  bootstrap intervals, all computed on the C runtime.
- A deterministic synthetic corpus, builders for WANLI and Wikispeedia, and
  importers for external fixtures.
- Trained models for the synthetic corpus, WANLI, Snake, Doom and the driving
  demo, with every reported number regenerable from `results/` and every
  checksum in `results/MANIFEST.json`.

**Demos**

- Blink plays Snake, Blink plays Doom and BlinkPilot, a driving demo: each runs
  in the browser on the WebAssembly build and headless in Node. They are
  published to GitHub Pages by `.github/workflows/pages.yml`, and
  `scripts/build_site.sh` assembles the same site locally and checks that
  every file the pages load is present.

**Tests and tooling**

- C unit tests for every kernel, the container parser, sessions and awkward
  inputs; parity tests between C, the float64 reference and PyTorch; and CI
  across GCC and clang, Linux and macOS, `-O0`, the sanitizers, SIMD compiled
  out, and each backend.
- `scripts/x86-docker-test.sh`: the x86-64 tests, parity and timings in a
  linux/amd64 container, for machines that are not x86-64.
- Latency and memory benchmarks (`make bench`), and comparisons with two
  external systems in `docs/RESULTS.md`.

**Licence**

- Apache License 2.0. Third-party datasets and fixtures keep their own terms;
  see `THIRD_PARTY.md`.

### Known limitations

- A 407k-parameter byte model learns the tasks it is trained on; it is not a
  language model. On natural language inference it is near chance, where a
  frozen 4B model is far ahead.
- Two of the four synthetic task families, `comparison` and `selection`, stay
  at chance in every run; `selection` needs binding between two positions
  inside the state, which the architecture does not yet provide.
- blink-small is worse than blink-tiny on every corpus measured.
- The x86-64 SIMD paths were verified for correctness under emulation and
  timed only on shared GitHub runners (AVX2 2.7× the scalar loop), not on a
  dedicated x86 machine. AVX-512 and VNNI are not used.
- The option list must be complete before scoring, and a probability is
  conditional on the options given.

[0.1.0]: https://github.com/sqliteai/blink/releases/tag/v0.1.0
