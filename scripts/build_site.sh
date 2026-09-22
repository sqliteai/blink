#!/usr/bin/env bash
# Assemble the static site GitHub Pages serves, into OUT (default: site/).
#
#   /                  BlinkPilot, one self-contained HTML file
#   /examples/snake/   Blink plays Snake
#   /examples/doom/    Blink plays Doom
#   /examples/pilot/   BlinkPilot, the unbundled page
#
# The pages load ../../wasm/blink.mjs, ../../build/wasm/blink.wasm and
# ../../artifacts/<model>.blink by relative path, so the site mirrors the
# repository layout for exactly those files. Every relative path a page or its
# modules reference is checked to exist in OUT before the script succeeds;
# that is what makes the site work from any base path, including a project
# page under /<repository>/.
#
# Needs build/wasm/blink.wasm (make wasm) and Node.
#
#   bash scripts/build_site.sh [OUT]
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-site}

[ -f build/wasm/blink.wasm ] || { echo "build/wasm/blink.wasm missing: run make wasm" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/examples" "$OUT/wasm" "$OUT/build/wasm" "$OUT/artifacts"

node examples/pilot/bundle.mjs artifacts/blink-tiny-pilot-s7.blink "$OUT/index.html" >/dev/null
for demo in snake doom pilot; do
  mkdir -p "$OUT/examples/$demo"
  cp examples/$demo/*.html examples/$demo/*.mjs "$OUT/examples/$demo/"
  cp artifacts/blink-tiny-$demo-s7.blink "$OUT/artifacts/"
done
# Generators and headless players are not needed by the pages.
rm -f "$OUT"/examples/*/make_data.mjs "$OUT"/examples/*/play.mjs \
      "$OUT"/examples/pilot/drive.mjs "$OUT"/examples/pilot/bundle.mjs
cp wasm/blink.mjs "$OUT/wasm/"
cp build/wasm/blink.wasm "$OUT/build/wasm/"
touch "$OUT/.nojekyll"

# Every relative reference must resolve inside the site.
node - "$OUT" <<'EOF'
const fs = require("node:fs");
const path = require("node:path");
const out = path.resolve(process.argv[2]);
const pattern = /(?:from\s*|import\(\s*|\?\?\s*|src=|href=)["'](\.{1,2}\/[^"'?#]+)["']/g;
const seen = new Set();
let missing = 0;
function check(file) {
  if (seen.has(file)) return;
  seen.add(file);
  const text = fs.readFileSync(file, "utf8");
  for (const match of text.matchAll(pattern)) {
    const target = path.resolve(path.dirname(file), match[1]);
    if (!target.startsWith(out) || !fs.existsSync(target)) {
      console.error(`missing: ${path.relative(out, file)} -> ${match[1]}`);
      missing++;
    } else if (/\.(mjs|js|html)$/.test(target)) {
      check(target);
    }
  }
}
for (const demo of ["snake", "doom", "pilot"]) {
  check(path.join(out, "examples", demo, "index.html"));
}
// The root page is the self-contained bundle: its wasm and model are inlined,
// and the default paths left in its text are never fetched.
if (missing) process.exit(1);
console.log(`site: ${seen.size} pages and modules, every relative reference resolves`);
EOF
du -sh "$OUT" | awk '{print "site: " $1 " in '"$OUT"'"}'
