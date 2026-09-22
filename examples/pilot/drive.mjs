/* drive.mjs -- Blink drives the Interstate 08 trip through the WebAssembly
 * runtime, headless.
 *
 *   node examples/pilot/drive.mjs MODEL.blink [--trips 20] [--density 1]
 *
 * Drives the same seeded trips with four policies and prints one table:
 *
 *   blink        argmax of blink_score over the six maneuvers
 *   oracle       the rule that labelled the training data (upper bound)
 *   random-safe  uniform over maneuvers that are neither illegal nor a crash
 *   random       uniform over all six maneuvers
 *
 * The menu is encoded once per run; each decision encodes a new state and
 * scores the fixed question against it.
 *
 * SPDX-License-Identifier: Apache-2.0 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadBlink } from "../../wasm/blink.mjs";
import { DECISION_S, OPTIONS, QUESTION, World, best, describe, oracle, rng } from "./world.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? Number(args[i + 1]) : fallback;
};
const modelPath = args.find((a) => a.endsWith(".blink"));
if (!modelPath) {
  console.error("usage: drive.mjs MODEL.blink [--trips N] [--density D]");
  process.exit(2);
}
const TRIPS = flag("--trips", 20);
const DENSITY = flag("--density", 1);
const MAX_SECONDS = 600;
const STEP = 1 / 20;

const blink = await loadBlink(readFileSync(join(here, "..", "..", "build", "wasm", "blink.wasm")));
const model = blink.openModel(readFileSync(modelPath));
const session = model.createSession({ maxOptions: OPTIONS.length });
session.setMenu(OPTIONS);

let stateSeconds = 0, scoreSeconds = 0, decisions = 0;

const POLICIES = {
  blink(outcomes, random) {
    const text = describe(outcomes, random);
    const t0 = performance.now();
    session.setState(text);
    const t1 = performance.now();
    const r = session.score(QUESTION);
    stateSeconds += (t1 - t0) / 1e3;
    scoreSeconds += (performance.now() - t1) / 1e3;
    ++decisions;
    return OPTIONS[r.argmax];
  },
  oracle: (outcomes) => oracle(outcomes),
  "random-safe"(outcomes, random) {
    const safe = OPTIONS.filter((o) => outcomes[o].rank <= 6);
    const pool = safe.length ? safe : OPTIONS;
    return pool[Math.floor(random() * pool.length)];
  },
  random: (outcomes, random) => OPTIONS[Math.floor(random() * OPTIONS.length)],
};

const table = [];
for (const [name, policy] of Object.entries(POLICIES)) {
  let arrived = 0, crashes = 0, violations = 0, agree = 0, turns = 0;
  const times = [];
  for (let g = 0; g < TRIPS; ++g) {
    const world = new World({ seed: 500009 + g, density: DENSITY });
    const random = rng(7 + g);
    let next = 0;
    while (!world.arrived && world.t < MAX_SECONDS) {
      if (world.t >= next) {
        next += DECISION_S;
        const outcomes = world.plan();
        const move = policy(outcomes, random);
        agree += best(outcomes).includes(move) ? 1 : 0;
        ++turns;
        world.decide(move);
      }
      world.step(STEP);
    }
    if (world.arrived) { ++arrived; times.push(world.t); }
    crashes += world.crashes;
    violations += world.violations;
  }
  times.sort((a, b) => a - b);
  table.push({
    policy: name,
    arrived: `${arrived}/${TRIPS}`,
    "median trip s": times.length ? times[times.length >> 1].toFixed(1) : "-",
    crashes,
    violations,
    "picks a best maneuver": (agree / turns).toFixed(4),
  });
}

const info = model.info();
console.log(`${modelPath}: ${info.name}, ${info.parameters} parameters, blink ${blink.version} (wasm)`);
console.log(`${TRIPS} trips Millbrook -> Cedar Town, traffic density ${DENSITY}, same seeds for every policy\n`);
console.table(table);
const us = (s) => (s / decisions * 1e6).toFixed(1);
console.log(`\nwasm, ${decisions} decisions: blink_state_set ${us(stateSeconds)} µs, ` +
  `blink_score ${us(scoreSeconds)} µs, ${Math.round(decisions / (stateSeconds + scoreSeconds))} decisions/s`);
