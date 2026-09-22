/* bundle.mjs -- build BlinkPilot as one self-contained HTML file.
 *
 *   node examples/pilot/bundle.mjs [MODEL.blink] [OUT.html]
 *
 * Inlines wasm/blink.mjs, world.mjs, render.mjs, build/wasm/blink.wasm and
 * the model into index.html. The result needs no server: open it straight
 * from disk (file://) or publish it on any static host. Defaults:
 * artifacts/blink-tiny-pilot-s7.blink -> build/blinkpilot.html.
 *
 * SPDX-License-Identifier: Apache-2.0 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const modelPath = process.argv[2] ?? join(root, "artifacts", "blink-tiny-pilot-s7.blink");
const out = process.argv[3] ?? join(root, "build", "blinkpilot.html");
const read = (p) => readFileSync(join(root, p), "utf8");

/* The modules share no top-level names, so they concatenate into one module
 * once their import lines and export keywords are gone. */
const strip = (src) => src
  .replace(/^import[\s\S]*?from\s+"[^"]+";\s*$/gm, "")
  .replace(/^export (?=(async |function|const|let|class))/gm, "");

const html = read("examples/pilot/index.html");
const open = '<script type="module">';
const start = html.indexOf(open), end = html.indexOf("</script>", start);
if (start < 0 || end < 0) throw new Error("index.html: inline module script not found");
const page = html.slice(start + open.length, end);

const embedded = {
  name: basename(modelPath),
  wasm: readFileSync(join(root, "build", "wasm", "blink.wasm")).toString("base64"),
  model: readFileSync(modelPath).toString("base64"),
};
const script = [
  `globalThis.BLINKPILOT_EMBEDDED = ${JSON.stringify(embedded)};`,
  strip(read("wasm/blink.mjs")),
  strip(read("examples/pilot/world.mjs")),
  strip(read("examples/pilot/render.mjs")),
  strip(page),
].join("\n");
if (/<\/script/i.test(script)) throw new Error("bundled script contains </script");

mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, html.slice(0, start + open.length) + "\n" + script + html.slice(end));
console.log(`${out}: ${(Buffer.byteLength(html) / 1024).toFixed(0)} KB page -> ` +
  `${(readFileSync(out).length / 1024).toFixed(0)} KB self-contained`);
