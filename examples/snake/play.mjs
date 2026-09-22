/* play.mjs -- Blink plays Snake through the WebAssembly runtime, headless.
 *
 *   node examples/snake/play.mjs MODEL.blink [--games 100] [--size 12] [--show]
 *
 * Plays the same seeded games with four policies and prints one table:
 *
 *   blink        argmax of blink_score over the four moves
 *   oracle       the rule that labelled the training data (upper bound)
 *   random-safe  uniform over moves that do not die immediately
 *   random       uniform over all four moves
 *
 * The menu is encoded once per game; each turn encodes a new state and scores
 * the fixed question against it.
 *
 * SPDX-License-Identifier: Apache-2.0 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadBlink } from "../../wasm/blink.mjs";
import { MOVES, QUESTION, Snake, rng } from "./snake.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? Number(args[i + 1]) : fallback;
};
const modelPath = args.find((a) => a.endsWith(".blink"));
if (!modelPath) {
  console.error("usage: play.mjs MODEL.blink [--games N] [--size N] [--show]");
  process.exit(2);
}
const GAMES = flag("--games", 100);
const SIZE = flag("--size", 12);
const SHOW = args.includes("--show");
const MAX_STEPS = 2000;
const MAX_HUNGRY = SIZE * SIZE * 2;

const blink = await loadBlink(readFileSync(join(here, "..", "..", "build", "wasm", "blink.wasm")));
const model = blink.openModel(readFileSync(modelPath));
const session = model.createSession({ maxOptions: 4 });
session.setMenu(MOVES);

let stateSeconds = 0, scoreSeconds = 0, decisions = 0;

const POLICIES = {
  blink(game, outcomes, random) {
    const text = game.describe(outcomes, random);
    const t0 = performance.now();
    session.setState(text);
    const t1 = performance.now();
    const r = session.score(QUESTION);
    stateSeconds += (t1 - t0) / 1e3;
    scoreSeconds += (performance.now() - t1) / 1e3;
    ++decisions;
    return MOVES[r.argmax];
  },
  oracle(game, outcomes, random) {
    const best = game.best(outcomes);
    return best[Math.floor(random() * best.length)];
  },
  "random-safe"(game, outcomes, random) {
    const safe = MOVES.filter((m) => !["wall", "body"].includes(outcomes[m].kind));
    const pool = safe.length ? safe : MOVES;
    return pool[Math.floor(random() * pool.length)];
  },
  random(game, outcomes, random) {
    return MOVES[Math.floor(random() * 4)];
  },
};

function draw(game) {
  const grid = Array.from({ length: game.height }, () => Array(game.width).fill("·"));
  if (game.food) grid[game.food[1]][game.food[0]] = "●";
  game.body.forEach(([x, y], i) => { grid[y][x] = i ? "▪" : "■"; });
  return grid.map((row) => row.join(" ")).join("\n");
}

const table = [];
for (const [name, policy] of Object.entries(POLICIES)) {
  const scores = [];
  const ends = {};
  let agree = 0, turns = 0;
  for (let g = 0; g < GAMES; ++g) {
    const game = new Snake({ width: SIZE, height: SIZE, seed: 100003 + g });
    const random = rng(7 + g);
    while (game.alive && game.steps < MAX_STEPS && game.sinceFood < MAX_HUNGRY) {
      const outcomes = game.outcomes();
      const move = policy(game, outcomes, random);
      agree += game.best(outcomes).includes(move) ? 1 : 0;
      ++turns;
      game.step(move);
    }
    const end = game.alive ? (game.steps >= MAX_STEPS ? "step cap" : "starved") : game.death;
    ends[end] = (ends[end] ?? 0) + 1;
    scores.push(game.score);
    if (SHOW && name === "blink" && g === 0) {
      console.log(`blink, game 0: score ${game.score}, ${game.steps} steps, ended by ${end}\n${draw(game)}\n`);
    }
  }
  scores.sort((a, b) => a - b);
  const mean = scores.reduce((s, v) => s + v, 0) / scores.length;
  table.push({
    policy: name,
    "mean food": mean.toFixed(2),
    median: scores[scores.length >> 1],
    best: scores[scores.length - 1],
    "oracle-agreement": (agree / turns).toFixed(4),
    endings: Object.entries(ends).map(([k, v]) => `${k} ${v}`).join(", "),
  });
}

const info = model.info();
console.log(`${modelPath}: ${info.name}, ${info.parameters} parameters, blink ${blink.version} (wasm)`);
console.log(`${GAMES} games on ${SIZE}x${SIZE}, same seeds for every policy\n`);
console.table(table);
const us = (s) => (s / decisions * 1e6).toFixed(1);
console.log(`\nwasm, ${decisions} decisions: blink_state_set ${us(stateSeconds)} µs, ` +
  `blink_score ${us(scoreSeconds)} µs, ${Math.round(decisions / (stateSeconds + scoreSeconds))} moves/s`);
