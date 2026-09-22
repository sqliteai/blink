# Blink -- embeddable one-pass typed-decision runtime.
#
#   make              build the static and shared libraries, tools, benchmarks
#   make test         run the C unit tests
#   make bench        run the latency and memory benchmarks
#   make all-checks   build, C tests, Python tests, accuracy, benchmarks
#   make wasm         build/wasm/blink.wasm, for browsers and Node
#   make test-wasm    check the wasm build against the native one
#   make pilot        build/blinkpilot.html, the driving demo in one file
#   make install      the blink tool, libblink, blink.h, blink.pc and the man
#                     page under PREFIX (default /usr/local); DESTDIR stages
#   make uninstall    remove what make install put there
#   make ACCELERATE=1 test
#                     macOS: the Accelerate backend, in build-accelerate/
#   make W8A8=1 test  int8 activations with SDOT/I8MM, in build-w8a8/
#
# -ffp-contract=off keeps a fused multiply-add from changing results between
# compilers and architectures, which is what makes the parity test meaningful.

CC      ?= cc
AR      ?= ar
CSTD    ?= -std=c99
OPT     ?= -O3
WARN    ?= -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
FPFLAGS ?= -ffp-contract=off -fno-fast-math
# -MMD -MP emits a .d file per object listing the headers it included, so
# editing a header rebuilds what depends on it. Without this, changing
# include/blink.h or src/blink_internal.h silently leaves stale objects linked
# -- which it did once, and the resulting binary reported the wrong error for
# an hour before anyone noticed.
DEPFLAGS ?= -MMD -MP
CFLAGS  ?= $(CSTD) $(OPT) $(WARN) $(FPFLAGS) $(DEPFLAGS) -Iinclude -Isrc
LDLIBS  ?= -lm

BUILD := build

# `make ACCELERATE=1` (macOS only) routes the projections through Accelerate's
# BLAS, which runs on the Apple-silicon matrix unit. It builds into its own
# directory so that the default build, which everything else in the repository
# reads, is never replaced by it. See the note in src/blink_kernels.c for what
# changes: fp32-rounding-level results and a dense fp32 copy of the
# projections held by the model.
ACCELERATE ?= 0
ifeq ($(ACCELERATE),1)
ifneq ($(shell uname -s),Darwin)
$(error ACCELERATE=1 needs macOS)
endif
BUILD   := build-accelerate
CFLAGS  += -DBLINK_ACCELERATE=1
CFLAGS  += -DBLINK_BUILD_DIR='"$(BUILD)"'
LDLIBS  += -framework Accelerate
endif

# `make W8A8=1` quantizes the activations of every projection to int8 as well
# and multiplies with SDOT, or with I8MM on non-Apple cores that have it (run
# time). Its own directory, like ACCELERATE, because it computes a slightly
# different function: see the W8A8 note in src/blink_internal.h.
W8A8 ?= 0
ifeq ($(W8A8),1)
ifeq ($(ACCELERATE),1)
$(error W8A8=1 and ACCELERATE=1 are alternative backends; pick one)
endif
BUILD   := build-w8a8
CFLAGS  += -DBLINK_W8A8=1
CFLAGS  += -DBLINK_BUILD_DIR='"$(BUILD)"'
endif

LIB   := $(BUILD)/libblink.a
SRC   := src/blink_kernels.c src/blink_crc32.c src/blink_model.c \
         src/blink_runtime.c src/blink_alloc.c
OBJ   := $(patsubst src/%.c,$(BUILD)/%.o,$(SRC))

# Release and binary-interface versions, read from include/blink.h so the
# header stays the only place they are written.
VERSION := $(shell awk '/\#define BLINK_VERSION_(MAJOR|MINOR|PATCH) /{v=v s $$3; s="."} END{print v}' include/blink.h)
ABI     := $(shell awk '/\#define BLINK_ABI_VERSION /{sub("u","",$$3); print $$3}' include/blink.h)

# The shared library is what the ctypes binding loads, so the accuracy and
# parity harnesses exercise the same code the C tests do. Its installed name
# carries the ABI version: libblink.so.1 (soname) on Linux, libblink.1.dylib
# on macOS, with an unversioned symlink for linking.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHLIB_EXT   := dylib
SHLIB_FLAGS := -dynamiclib -install_name @rpath/libblink.$(ABI).dylib \
               -current_version $(VERSION) -compatibility_version $(ABI) \
               -Wl,-headerpad_max_install_names
else
SHLIB_EXT   := so
SHLIB_FLAGS := -shared -Wl,-soname,libblink.so.$(ABI)
endif
SHLIB := $(BUILD)/libblink.$(SHLIB_EXT)

# Same flags with the SIMD kernel compiled out, used by the `scalar` target.
SCALAR_CFLAGS := $(CSTD) $(OPT) $(WARN) $(FPFLAGS) -DBLINK_SCALAR_ONLY=1 \
                 -Iinclude -Isrc

TEST_SRC  := $(wildcard tests/c/test_*.c)
TEST_BIN  := $(patsubst tests/c/%.c,$(BUILD)/%,$(TEST_SRC))
TOOL_SRC  := $(wildcard tools/*.c)
TOOL_BIN  := $(patsubst tools/%.c,$(BUILD)/%,$(TOOL_SRC))
BENCH_SRC := $(wildcard bench/*.c)
BENCH_BIN := $(patsubst bench/%.c,$(BUILD)/%,$(BENCH_SRC))
EX_SRC    := $(wildcard examples/*.c)
EX_BIN    := $(patsubst examples/%.c,$(BUILD)/%,$(EX_SRC))

# Synthesised containers. The C tool builds them, so the C suite and the
# benchmarks need neither Python nor a trained checkpoint.
FIXTURES := $(BUILD)/synth-nano.blink $(BUILD)/synth-tiny.blink \
            $(BUILD)/synth-small.blink

.PHONY: all test bench bench-save fixtures tools clean all-checks scalar examples

all: $(LIB) $(SHLIB) $(TOOL_BIN) $(BENCH_BIN) $(EX_BIN)

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJ)
	$(AR) rcs $@ $(OBJ)

$(SHLIB): $(SRC) Makefile | $(BUILD)
	$(CC) $(CFLAGS) -fPIC $(SHLIB_FLAGS) $(SRC) $(LDLIBS) -o $@

$(BUILD)/%: tests/c/%.c $(LIB) tests/c/harness.h | $(BUILD)
	$(CC) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD)/%: tools/%.c $(LIB) | $(BUILD)
	$(CC) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD)/%: bench/%.c $(LIB) | $(BUILD)
	$(CC) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD)/%: examples/%.c $(LIB) | $(BUILD)
	$(CC) $(CFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD)/synth-%.blink: $(BUILD)/blink_synth
	@$< $@ $* >/dev/null

fixtures: $(FIXTURES)

tools: $(TOOL_BIN)

examples: $(EX_BIN) $(FIXTURES)
	@$(BUILD)/embed_static $(BUILD)/synth-tiny.blink

test: $(TEST_BIN) $(FIXTURES)
	@bash tests/c/check_no_malloc.sh $(BUILD)
	@fail=0; \
	for binary in $(TEST_BIN); do \
	  printf '%-28s ' "$$(basename $$binary)"; \
	  if $$binary >$(BUILD)/$$(basename $$binary).log 2>&1; then \
	    echo "ok   $$(tail -1 $(BUILD)/$$(basename $$binary).log)"; \
	  else \
	    echo "FAIL"; cat $(BUILD)/$$(basename $$binary).log; fail=1; \
	  fi; \
	done; \
	exit $$fail

bench: $(BENCH_BIN) $(FIXTURES)
	@for binary in $(BENCH_BIN); do $$binary; done

# Rebuild with the SIMD kernel disabled, to reproduce the scalar/NEON
# comparison in docs/RESULTS.md. The scalar loop is the normative definition;
# tests/c/test_kernels.c checks that the two agree.
scalar:
	$(MAKE) clean
	$(MAKE) CFLAGS='$(SCALAR_CFLAGS)' all fixtures

# Write the benchmark results that docs/RESULTS.md sections 1 and 2 are
# generated from. Run it on an idle machine.
bench-save: $(BENCH_BIN) $(FIXTURES)
	@mkdir -p results
	./build/bench_latency > results/bench-latency.jsonl
	./build/bench_memory  > results/bench-memory.jsonl
	$(MAKE) scalar
	./build/bench_latency > results/bench-latency-scalar.jsonl
	$(MAKE) clean && $(MAKE) && $(MAKE) fixtures
	@echo "wrote results/bench-*.jsonl; regenerate the document with"
	@echo "  python scripts/report_results.py --write"

all-checks:
	@bash scripts/run_all.sh

# WebAssembly build: the same five sources, compiled for wasm32-wasip1 as a
# reactor (no main) with the scalar kernel: the SIMD paths are NEON and x86-64.
# Needs a clang with the wasm32 target, wasm-ld, wasi-libc and the wasm
# compiler-rt builtins. On macOS: brew install lld wasi-libc wasi-runtimes.
# The browser loads weights with blink_model_open_memory; the mmap path is
# still linked (through wasi-emulated-mman) but unused there.
WASM_CC      ?= $(firstword $(wildcard /opt/homebrew/opt/llvm/bin/clang /usr/local/opt/llvm/bin/clang) clang)
WASM_LD      ?= $(firstword $(wildcard /opt/homebrew/opt/lld/bin/wasm-ld /usr/local/opt/lld/bin/wasm-ld) wasm-ld)
WASI_SYSROOT ?= $(firstword $(wildcard /opt/homebrew/share/wasi-sysroot /usr/local/share/wasi-sysroot /opt/wasi-sdk/share/wasi-sysroot))
WASI_RT      ?= $(firstword $(wildcard /opt/homebrew/share/wasi-runtimes/lib/wasm32-unknown-wasip1 /opt/homebrew/Cellar/wasi-runtimes/*/share/wasi-runtimes/lib/wasm32-unknown-wasip1))
comma := ,
WASM_EXPORTS := blink_status_string blink_version blink_backend blink_model_open_memory \
                blink_model_close blink_model_get_info blink_session_size \
                blink_session_init blink_session_create blink_session_free \
                blink_state_set blink_menu_set blink_score blink_score_batch \
                blink_decide \
                blink_last_logits blink_last_attention malloc free
WASM_CFLAGS  := --target=wasm32-wasip1 --sysroot=$(WASI_SYSROOT) $(CSTD) $(OPT) \
                $(WARN) $(FPFLAGS) -D_POSIX_C_SOURCE=200809L \
                -D_WASI_EMULATED_MMAN -Iinclude -Isrc
WASM_LDFLAGS := -mexec-model=reactor -fuse-ld=$(WASM_LD) \
                -Wl,--initial-memory=33554432 -Wl,--max-memory=268435456 \
                -Wl,--growable-table $(addprefix -Wl$(comma)--export=,$(WASM_EXPORTS)) \
                -lwasi-emulated-mman $(if $(WASI_RT),-L$(WASI_RT) -lclang_rt.builtins)
WASM := $(BUILD)/wasm/blink.wasm

.PHONY: wasm test-wasm
wasm: $(WASM)

# The wasm build against the native one on the synthesised fixtures. Needs
# Node >= 18.
test-wasm: $(WASM) $(FIXTURES) $(BUILD)/blink
	@node tests/wasm/test_wasm.mjs

$(WASM): $(SRC) include/blink.h src/blink_internal.h
	@mkdir -p $(dir $@)
	$(WASM_CC) $(WASM_CFLAGS) $(WASM_LDFLAGS) $(SRC) -o $@

# BlinkPilot as a single HTML file: page, modules, wasm and model inlined, so
# it opens from disk or any static host with no server. Needs Node.
PILOT_MODEL ?= artifacts/blink-tiny-pilot-s7.blink
PILOT_SRC   := $(wildcard examples/pilot/*.mjs examples/pilot/*.html) wasm/blink.mjs

.PHONY: pilot
pilot: $(BUILD)/blinkpilot.html

$(BUILD)/blinkpilot.html: $(WASM) $(PILOT_MODEL) $(PILOT_SRC)
	@node examples/pilot/bundle.mjs $(PILOT_MODEL) $@

# ---------------------------------------------------------------- install
#
#   make install                        # under /usr/local
#   make install PREFIX=$HOME/.local
#   make install DESTDIR=/tmp/stage     # staged, for packaging
#
# The blink tool links libblink statically, so it runs without the shared
# library. `make W8A8=1 install` or `make ACCELERATE=1 install` installs that
# variant instead; blink --version says which one is installed.
PREFIX      ?= /usr/local
BINDIR      ?= $(PREFIX)/bin
LIBDIR      ?= $(PREFIX)/lib
INCLUDEDIR  ?= $(PREFIX)/include
MANDIR      ?= $(PREFIX)/share/man
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
INSTALL     ?= install
PC_LIBS_PRIVATE := -lm$(if $(filter 1,$(ACCELERATE)), -framework Accelerate)

.PHONY: install uninstall
install: $(BUILD)/blink $(LIB) $(SHLIB)
	$(INSTALL) -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(LIBDIR) \
	    $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(MANDIR)/man1 $(DESTDIR)$(PKGCONFIGDIR)
	$(INSTALL) -m 755 $(BUILD)/blink $(DESTDIR)$(BINDIR)/blink
	$(INSTALL) -m 644 $(LIB) $(DESTDIR)$(LIBDIR)/libblink.a
	$(INSTALL) -m 644 include/blink.h $(DESTDIR)$(INCLUDEDIR)/blink.h
	$(INSTALL) -m 644 docs/blink.1 $(DESTDIR)$(MANDIR)/man1/blink.1
ifeq ($(UNAME_S),Darwin)
	$(INSTALL) -m 755 $(SHLIB) $(DESTDIR)$(LIBDIR)/libblink.$(ABI).dylib
	install_name_tool -id $(LIBDIR)/libblink.$(ABI).dylib \
	    $(DESTDIR)$(LIBDIR)/libblink.$(ABI).dylib
	codesign --force --sign - $(DESTDIR)$(LIBDIR)/libblink.$(ABI).dylib 2>/dev/null || true
	ln -sf libblink.$(ABI).dylib $(DESTDIR)$(LIBDIR)/libblink.dylib
else
	$(INSTALL) -m 755 $(SHLIB) $(DESTDIR)$(LIBDIR)/libblink.so.$(VERSION)
	ln -sf libblink.so.$(VERSION) $(DESTDIR)$(LIBDIR)/libblink.so.$(ABI)
	ln -sf libblink.so.$(ABI) $(DESTDIR)$(LIBDIR)/libblink.so
endif
	printf '%s\n' 'prefix=$(PREFIX)' 'libdir=$(LIBDIR)' 'includedir=$(INCLUDEDIR)' '' \
	    'Name: blink' 'Description: One-pass typed-decision runtime' \
	    'URL: https://github.com/sqliteai/blink' \
	    'Version: $(VERSION)' 'Libs: -L$${libdir} -lblink' \
	    'Libs.private: $(PC_LIBS_PRIVATE)' 'Cflags: -I$${includedir}' \
	    > $(DESTDIR)$(PKGCONFIGDIR)/blink.pc
	@echo "installed blink $(VERSION) under $(DESTDIR)$(PREFIX)"

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/blink $(DESTDIR)$(LIBDIR)/libblink.a \
	    $(DESTDIR)$(LIBDIR)/libblink.$(ABI).dylib $(DESTDIR)$(LIBDIR)/libblink.dylib \
	    $(DESTDIR)$(LIBDIR)/libblink.so.$(VERSION) $(DESTDIR)$(LIBDIR)/libblink.so.$(ABI) \
	    $(DESTDIR)$(LIBDIR)/libblink.so $(DESTDIR)$(INCLUDEDIR)/blink.h \
	    $(DESTDIR)$(MANDIR)/man1/blink.1 $(DESTDIR)$(PKGCONFIGDIR)/blink.pc


clean:
	rm -rf $(BUILD)

-include $(OBJ:.o=.d)
-include $(TEST_BIN:=.d)
-include $(TOOL_BIN:=.d)
-include $(BENCH_BIN:=.d)
-include $(EX_BIN:=.d)
