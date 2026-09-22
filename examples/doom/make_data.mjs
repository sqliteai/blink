/* make_data.mjs -- write a ViZDoom decision corpus in the Blink JSONL format.
 *
 *   node examples/doom/make_data.mjs data/doom
 *
 * Both scenarios go into one corpus, so one model plays both. Episodes are played by the teacher with a share of
 * random actions, so the corpus also covers states a weaker player reaches.
 * Every decision is one row; splits are by episode (`group`), never by row.
 *
 * Seeds: play.mjs evaluates on the held-out test and OOD seeds, so nothing
 * here touches them. Training seeds come from a separate range; validation
 * and test rows use the dev and calibration seed ranges.
 *
 * Predict Position has one rocket. Rows after it is fired are dropped --
 * nothing the policy does then changes the outcome -- and exploration never
 * fires at random, which would spend the episode on its first step.
 *
 * Deterministic: the same command writes the same bytes.
 *
 * SPDX-License-Identifier: Apache-2.0 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { ACTIONS, Doom, OPTIONS, QUESTION, rng } from "./doom.mjs";

const EPSILON = 0.2;
const SPLITS = {
  train: { basic: [1_000_000, 3000], predict_position: [2_000_000, 4000] },
  validation: { basic: [9_010_000, 64], predict_position: [9_300_512, 64] },
  test: { basic: [9_020_000, 64], predict_position: [9_300_576, 64] },
};

function play(scenario, first, episodes, random) {
  const rows = [];
  let wins = 0;
  for (let seed = first; seed < first + episodes; ++seed) {
    const game = new Doom({ scenario, seed, frameSkip: 4, maxSteps: 75 });
    while (!game.done) {
      const label = game.teacher();
      if (game.ammo > 0) {
        rows.push({
          state: game.describe(),
          question: QUESTION,
          options: OPTIONS[scenario],
          label: ACTIONS.indexOf(label),
          group: `${scenario}-${seed}`,
          family: scenario,
        });
      }
      let move = label;
      if (random() < EPSILON) {
        const pool = scenario === "basic" ? ACTIONS : ["left", "right", "noop"];
        move = pool[Math.floor(random() * pool.length)];
      }
      game.step(move);
    }
    wins += game.success;
  }
  return { rows, wins };
}

const out = process.argv[2] ?? "data/doom";
mkdirSync(out, { recursive: true });
const manifest = { generator: "examples/doom/make_data.mjs", epsilon: EPSILON,
  frameSkip: 4, maxSteps: 75, splits: {} };
for (const [split, scenarios] of Object.entries(SPLITS)) {
  const random = rng(split.length * 7919);
  let rows = [];
  for (const [scenario, [first, episodes]] of Object.entries(scenarios)) {
    const played = play(scenario, first, episodes, random);
    rows = rows.concat(played.rows);
    manifest.splits[`${split}/${scenario}`] = { firstSeed: first, episodes,
      rows: played.rows.length, teacherWinsWithExploration: played.wins };
    console.log(`${split}/${scenario}: ${episodes} episodes, ${played.rows.length} rows, ` +
      `${played.wins} wins with exploration`);
  }
  /* Interleave the scenarios so a batch is never all one of them. */
  for (let i = rows.length - 1; i > 0; --i) {
    const j = Math.floor(random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  writeFileSync(join(out, `${split}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
}
writeFileSync(join(out, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
