/* play.mjs -- Blink plays Doom through the WebAssembly runtime, headless.
 *
 *   node examples/doom/play.mjs MODEL.blink [--episodes 128] [--json FILE]
 *
 * Plays the held-out cases with every policy and prints one table:
 *
 *   scenario           split  ticks/action  decisions  seeds
 *   basic              test   4             75         9030000 .. 9030127
 *   basic              ood    8             38         9040000 .. 9040127
 *   predict_position   test   4             75         9300640 .. 9300767
 *   predict_position   ood    8             38         9300768 .. 9300895
 *
 * Policies, all fed the same observations:
 *
 *   blink        argmax of blink_score over the four actions
 *   blink e=0.1  the argmax, replaced by a uniformly
 *                random action with probability 0.1 (one generator, seed 17)
 *   teacher      the rule that labelled the training data. In Predict
 *                Position it reads world positions Blink never sees
 *   random       uniform over the four actions
 *
 * The menu is encoded once per scenario; each decision encodes a new state
 * and scores the fixed question against it.
 *
 * SPDX-License-Identifier: Apache-2.0 */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadBlink } from "../../wasm/blink.mjs";
import { ACTIONS, Doom, OPTIONS, QUESTION, rng } from "./doom.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const modelPath = args.find((a) => a.endsWith(".blink"));
if (!modelPath) {
  console.error("usage: play.mjs MODEL.blink [--episodes N] [--json FILE]");
  process.exit(2);
}
const EPISODES = Number(option("--episodes", 128));

const CASES = [
  { scenario: "basic", split: "test", frameSkip: 4, maxSteps: 75, first: 9_030_000 },
  { scenario: "basic", split: "ood", frameSkip: 8, maxSteps: 38, first: 9_040_000 },
  { scenario: "predict_position", split: "test", frameSkip: 4, maxSteps: 75, first: 9_300_640 },
  { scenario: "predict_position", split: "ood", frameSkip: 8, maxSteps: 38, first: 9_300_768 },
];

/* Jev's published results on the same cases. They come from real ViZDoom;
 * these runs use doom.mjs. */
const PUBLISHED = {
  "basic/test": { Jev: 56 },
  "basic/ood": { Jev: 59 },
  "predict_position/test": { Jev: 11 },
  "predict_position/ood": { Jev: 8 },
};

const blink = await loadBlink(readFileSync(join(here, "..", "..", "build", "wasm", "blink.wasm")));
const model = blink.openModel(readFileSync(modelPath));
const session = model.createSession({ maxOptions: 4 });
let menu = null;
let stateSeconds = 0, scoreSeconds = 0, decisions = 0;

function blinkPolicy(epsilon) {
  const random = rng(17);
  return (game) => {
    if (menu !== game.scenario) {
      session.setMenu(OPTIONS[game.scenario]);
      menu = game.scenario;
    }
    const t0 = performance.now();
    session.setState(game.describe());
    const t1 = performance.now();
    const r = session.score(QUESTION);
    stateSeconds += (t1 - t0) / 1e3;
    scoreSeconds += (performance.now() - t1) / 1e3;
    ++decisions;
    if (epsilon && random() < epsilon) return ACTIONS[Math.floor(random() * 4)];
    return ACTIONS[r.argmax];
  };
}

const POLICIES = {
  blink: blinkPolicy(0),
  "blink e=0.1": blinkPolicy(0.1),
  teacher: (game) => game.teacher(),
  random: ((random) => () => ACTIONS[Math.floor(random() * 4)])(rng(17)),
};

const results = [];
const table = [];
for (const c of CASES) {
  const row = { case: `${c.scenario} ${c.split} (${c.frameSkip} ticks)` };
  for (const [name, policy] of Object.entries(POLICIES)) {
    let wins = 0, ticks = 0, agree = 0, turns = 0;
    for (let seed = c.first; seed < c.first + EPISODES; ++seed) {
      const game = new Doom({ scenario: c.scenario, seed, frameSkip: c.frameSkip, maxSteps: c.maxSteps });
      while (!game.done) {
        const action = policy(game);
        if (game.ammo > 0) {
          agree += action === game.teacher() ? 1 : 0;
          ++turns;
        }
        game.step(action);
      }
      if (game.success) {
        ++wins;
        ticks += game.ticks;
      }
    }
    row[name] = `${wins}/${EPISODES}`;
    results.push({ ...c, policy: name, wins, episodes: EPISODES,
      meanTicksToKill: wins ? +(ticks / wins).toFixed(1) : null,
      teacherAgreement: +(agree / turns).toFixed(4) });
  }
  const published = PUBLISHED[`${c.scenario}/${c.split}`];
  row["Jev (ViZDoom)"] = `${published.Jev}`;
  table.push(row);
}

const info = model.info();
console.log(`${modelPath}: ${info.name}, ${info.parameters} parameters, blink ${blink.version} (wasm)`);
console.log(`${EPISODES} episodes per case, held-out seeds, same seeds for every policy\n`);
console.table(table);
const us = (s) => (s / decisions * 1e6).toFixed(1);
console.log(`\nwasm, ${decisions} decisions: blink_state_set ${us(stateSeconds)} µs, ` +
  `blink_score ${us(scoreSeconds)} µs`);

const jsonPath = option("--json", null);
if (jsonPath) {
  writeFileSync(jsonPath, JSON.stringify({
    model: modelPath.split("/").pop(), parameters: info.parameters, episodes: EPISODES,
    results, published: PUBLISHED,
    timingMicroseconds: { stateSet: +us(stateSeconds), score: +us(scoreSeconds) },
  }, null, 2) + "\n");
}
