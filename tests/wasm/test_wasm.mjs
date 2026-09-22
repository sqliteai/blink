/* test_wasm.mjs -- the WebAssembly build must agree with the native one.
 *
 *   node tests/wasm/test_wasm.mjs [MODEL.blink ...]
 *
 * For each container (default: the three synthesised fixtures) it scores a
 * fixed set of decisions through build/wasm/blink.wasm and through the native
 * build/blink, and checks:
 *
 *   - same argmax, probabilities within 2e-5 (the CLI prints six decimals,
 *     and the native build may use the NEON kernel while wasm is scalar);
 *   - reusing an encoded state is bit-identical to re-encoding it, inside
 *     wasm, compared on the raw logits;
 *   - a corrupted container is refused with the checksum status.
 *
 * Needs `make wasm fixtures` first. SPDX-License-Identifier: Apache-2.0 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadBlink, BlinkError } from "../../wasm/blink.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const models = process.argv.length > 2 ? process.argv.slice(2)
  : ["nano", "tiny", "small"].map((p) => join(root, "build", `synth-${p}.blink`));

const CASES = [
  {
    state: "The parcel left the depot on Monday and has not arrived.",
    question: "Which team should handle this?",
    options: ["billing and payments", "delivery and logistics",
      "account access and sign-in", "device hardware repair"],
  },
  {
    state: "The order for Kofi's projector was cancelled on Tuesday morning.",
    question: "Assess the claim: the order was approved.",
    options: ["the evidence contradicts the claim",
      "the evidence is insufficient to decide", "the evidence supports the claim"],
  },
  {
    state: "Snake head at column 4, row 7, moving up. Food at column 9, row 2. " +
      "Wall directly above. Body to the left.",
    question: "Which move keeps the snake alive and brings it closer to the food?",
    options: ["up", "down", "left", "right"],
  },
  {
    state: "Città: Torino. Temperatura 31 °C, umidità 80%. ☀️",
    question: "Che tempo fa?",
    options: ["caldo e afoso", "freddo e secco"],
  },
];

let checks = 0, failures = 0;
/* Inputs over the model's limits are refused, not truncated, so the cases are
 * clipped to each model's geometry -- on a UTF-8 boundary -- before either
 * build sees them. */
function clip(text, maxBytes) {
  let out = "", used = 0;
  for (const ch of text) {
    const n = Buffer.byteLength(ch);
    if (used + n > maxBytes) break;
    out += ch; used += n;
  }
  return out;
}

function check(ok, what) {
  ++checks;
  if (!ok) { ++failures; console.error(`FAIL ${what}`); }
}

const blink = await loadBlink(readFileSync(join(root, "build", "wasm", "blink.wasm")));
const cli = join(root, "build", "blink");

for (const path of models) {
  const bytes = readFileSync(path);
  const model = blink.openModel(bytes);
  const session = model.createSession({ maxOptions: 8 });
  let worst = 0;

  const info = model.info();
  for (const raw of CASES) {
    const c = {
      state: clip(raw.state, info.maxState),
      question: clip(raw.question, info.maxQuestion),
      options: raw.options.map((o) => clip(o, info.maxOption)),
    };
    session.setState(c.state);
    session.setMenu(c.options);
    const wasm = session.score(c.question);
    const logitsReused = session.logits();

    const args = [path, "--state", c.state, "--question", c.question];
    for (const o of c.options) args.push("--option", o);
    const native = JSON.parse(execFileSync(cli, args, { encoding: "utf8" }));

    check(wasm.argmax === native.argmax,
      `${path}: argmax ${wasm.argmax} != native ${native.argmax} for "${c.question}"`);
    native.options.forEach((o, i) => {
      const diff = Math.abs(o.probability - wasm.probabilities[i]);
      worst = Math.max(worst, diff);
      check(diff <= 2e-5, `${path}: option ${i} differs by ${diff}`);
    });

    /* Re-encode the same state from scratch: must be bit-identical. */
    session.setState(c.state);
    session.score(c.question);
    const logitsFresh = session.logits();
    check(logitsReused.every((v, i) => Object.is(v, logitsFresh[i])),
      `${path}: re-encoded state is not bit-identical`);
  }

  session.free();
  model.close();
  console.log(`${path.split("/").pop().padEnd(22)} max |Δp| vs native = ${worst.toExponential(2)}`);

  /* Flip one byte of weight data: the checksum must refuse it. */
  const corrupt = Uint8Array.from(bytes);
  corrupt[corrupt.length - 1] ^= 0x5a;
  let status = 0;
  try { blink.openModel(corrupt).close(); } catch (e) {
    if (e instanceof BlinkError) status = e.status;
  }
  check(status === 5, `${path}: corrupted container gave status ${status}, want 5 (checksum)`);
}

console.log(`${checks} checks, ${failures} failures`);
process.exit(failures ? 1 : 0);
