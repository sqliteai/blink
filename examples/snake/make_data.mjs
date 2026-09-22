/* make_data.mjs -- write a Snake decision corpus in the Blink JSONL format.
 *
 *   node examples/snake/make_data.mjs data/snake
 *
 * Games are played by the oracle with a share of random non-lethal moves, so
 * the corpus also covers the states a weaker player reaches. Every turn is one
 * row. Splits are by game (`group`), never by row, and use disjoint seeds.
 *
 * Deterministic: the same command writes the same bytes.
 *
 * SPDX-License-Identifier: Apache-2.0 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { MOVES, QUESTION, Snake, rng } from "./snake.mjs";

const EPSILON = 0.2;         /* share of random non-lethal moves */
const MAX_STEPS = 600;
const MAX_HUNGRY = 150;      /* end a game that circles without eating */

function playGames(games, seed, split) {
  const rows = [];
  const random = rng(seed);
  for (let g = 0; g < games; ++g) {
    const size = 8 + Math.floor(random() * 7);       /* 8x8 .. 14x14 */
    const game = new Snake({ width: size, height: size, seed: seed * 7919 + g });
    while (game.alive && game.steps < MAX_STEPS && game.sinceFood < MAX_HUNGRY) {
      const outcomes = game.outcomes();
      const best = game.best(outcomes);
      const options = [...MOVES];
      for (let i = options.length - 1; i > 0; --i) {
        const j = Math.floor(random() * (i + 1));
        [options[i], options[j]] = [options[j], options[i]];
      }
      const label = best[Math.floor(random() * best.length)];
      rows.push({
        state: game.describe(outcomes, random),
        question: QUESTION,
        options,
        label: options.indexOf(label),
        group: `snake-${split}-${g}`,
        family: "snake",
      });

      let move = label;
      if (random() < EPSILON) {
        const safe = MOVES.filter((m) => !["wall", "body"].includes(outcomes[m].kind));
        if (safe.length) move = safe[Math.floor(random() * safe.length)];
      }
      game.step(move);
    }
  }
  return rows;
}

const out = process.argv[2] ?? "data/snake";
mkdirSync(out, { recursive: true });
const splits = { train: [150, 11], validation: [20, 23], test: [20, 37] };
const manifest = { generator: "examples/snake/make_data.mjs", epsilon: EPSILON, splits: {} };
for (const [name, [games, seed]] of Object.entries(splits)) {
  const rows = playGames(games, seed, name);
  writeFileSync(join(out, `${name}.jsonl`), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  manifest.splits[name] = { games, seed, rows: rows.length };
  console.log(`${name}: ${games} games, ${rows.length} rows`);
}
writeFileSync(join(out, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
