/* make_data.mjs -- write a driving decision corpus in the Blink JSONL format.
 *
 *   node examples/pilot/make_data.mjs data/pilot
 *
 * Trips are driven by the oracle with a share of random maneuvers, so the
 * corpus also covers the states a weaker driver reaches: tailgating, late
 * braking, a missed gap. Half the trips start somewhere along the route at a
 * random speed, so the interstate and the exit are not outnumbered by the
 * town. Every decision is one row. Splits are by trip (`group`), never by row,
 * and use disjoint seeds.
 *
 * Deterministic: the same command writes the same bytes.
 *
 * SPDX-License-Identifier: Apache-2.0 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  DECISION_S, MARKS, OPTIONS, QUESTION, World, best, describe, laneKind, lanesAt,
  limitAt, rng,
} from "./world.mjs";

const EPSILON = 0.2;        /* share of random maneuvers that are not disasters */
const CHAOS = 0.02;         /* share of uniformly random maneuvers              */
const MAX_SECONDS = 320;
/* Lane changes and slowing down are rare decisions. Each one is written again
 * this many extra times, with fresh phrasing and sentence order. Training
 * split only: validation and test keep the natural mix. */
const REPEAT = { left: 8, right: 3, slower: 3 };
const STEP = 1 / 20;

function drive(trips, seed, split) {
  const rows = [];
  const random = rng(seed);
  const maxBytes = { value: 0 };
  for (let g = 0; g < trips; ++g) {
    let start = 8, speed = 0;
    if (random() < 0.5) {
      start = 20 + random() * (MARKS.destination - 120);
      speed = limitAt(start) * random();
    }
    const lanes = lanesAt(start).filter((l) => laneKind(start, l) === "drive");
    const world = new World({
      seed: seed * 7919 + g,
      density: 0.4 + random(),
      start, speed,
      lane: lanes[Math.floor(random() * lanes.length)],
    });
    let next = 0;
    while (!world.arrived && world.t < MAX_SECONDS) {
      if (world.t >= next) {
        next += DECISION_S;
        const outcomes = world.plan();
        const top = best(outcomes);
        const label = top[Math.floor(random() * top.length)];
        const copies = split === "train" ? REPEAT[label] ?? 0 : 0;
        for (let copy = 0; copy <= copies; ++copy) {
          const options = [...OPTIONS];
          for (let i = options.length - 1; i > 0; --i) {
            const j = Math.floor(random() * (i + 1));
            [options[i], options[j]] = [options[j], options[i]];
          }
          const state = describe(outcomes, random);
          maxBytes.value = Math.max(maxBytes.value, Buffer.byteLength(state));
          rows.push({
            state, question: QUESTION, options,
            label: options.indexOf(label),
            group: `pilot-${split}-${g}`,
            family: "pilot",
          });
        }

        let move = label;
        const roll = random();
        if (roll < CHAOS) {
          move = OPTIONS[Math.floor(random() * OPTIONS.length)];
        } else if (roll < CHAOS + EPSILON) {
          const tolerable = OPTIONS.filter((o) => outcomes[o].rank <= 6);
          if (tolerable.length) move = tolerable[Math.floor(random() * tolerable.length)];
        }
        world.decide(move);
      }
      world.step(STEP);
    }
  }
  return { rows, maxBytes: maxBytes.value };
}

const out = process.argv[2] ?? "data/pilot";
mkdirSync(out, { recursive: true });
const splits = { train: [80, 11], validation: [10, 23], test: [10, 37] };
const manifest = {
  generator: "examples/pilot/make_data.mjs", epsilon: EPSILON, chaos: CHAOS, repeat: REPEAT,
  splits: {},
};
for (const [name, [trips, seed]] of Object.entries(splits)) {
  const { rows, maxBytes } = drive(trips, seed, name);
  writeFileSync(join(out, `${name}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const labels = {};
  for (const r of rows) labels[r.options[r.label]] = (labels[r.options[r.label]] ?? 0) + 1;
  manifest.splits[name] = { trips, seed, rows: rows.length, max_state_bytes: maxBytes, labels };
  console.log(`${name}: ${trips} trips, ${rows.length} rows, longest state ${maxBytes} bytes`, labels);
}
writeFileSync(join(out, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
