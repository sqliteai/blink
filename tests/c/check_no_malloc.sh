#!/usr/bin/env bash
# The scoring path must never allocate. blink_runtime.o holds the encoder, the
# head and the arena; blink_kernels.o holds the numeric loops. Neither may
# reference an allocator symbol. Allocation is confined to blink_alloc.o (the
# optional blink_session_create helper) and blink_model.o (model open time).
set -euo pipefail

build="${1:-build}"
pattern='_?(malloc|calloc|realloc|free|posix_memalign|aligned_alloc|valloc|reallocf|strdup)$'
status=0

for object in "$build/blink_runtime.o" "$build/blink_kernels.o"; do
  if [ ! -f "$object" ]; then
    echo "check_no_malloc: missing $object" >&2
    exit 2
  fi
  found=$(nm -u "$object" | awk '{print $NF}' | grep -E "$pattern" || true)
  if [ -n "$found" ]; then
    echo "check_no_malloc: $object references an allocator:" >&2
    echo "$found" >&2
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  echo "check_no_malloc               ok   scoring path references no allocator"
fi
exit "$status"
