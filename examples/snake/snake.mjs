/* snake.mjs -- the game, its text rendering and the oracle that labels it.
 *
 * One module serves both sides: make_data.mjs uses it to write the training
 * corpus and play.mjs / index.html use it to run the game against the model.
 * The state text the model is trained on and the state text it plays from are
 * therefore produced by the same function, which is the point.
 *
 * What the model sees is not the board. Each turn the game describes, in
 * shuffled order, what each of the four moves would lead to ("Going left
 * bites the body.", "Going up is open and heads toward the food."). The model
 * has to find the sentence about each option in the state and weigh the
 * outcomes against each other. Nothing in the text gives the ranking.
 *
 * SPDX-License-Identifier: Apache-2.0 */

export const MOVES = ["up", "down", "left", "right"];
const DELTA = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };

export const QUESTION = "Which move keeps the snake alive and gets it to the food?";

/* Outcomes, best first. The oracle picks the lowest rank. */
export const OUTCOMES = ["food", "toward", "away", "trap", "wall", "body"];
const RANK = Object.fromEntries(OUTCOMES.map((o, i) => [o, i]));

const PHRASES = {
  food: ["eats the food", "reaches the food", "lands on the food"],
  toward: ["is open and heads toward the food", "is free and gets closer to the food",
    "is clear and moves nearer the food"],
  away: ["is open but heads away from the food", "is free but moves away from the food",
    "is clear but gets farther from the food"],
  trap: ["leads into a dead end", "enters a pocket too small to escape",
    "boxes the snake in"],
  wall: ["hits the wall", "crashes into the wall", "runs into the edge"],
  body: ["bites the body", "runs into its own body", "crashes into the tail"],
};
const LEADS = ["Going", "Moving", "Turning"];

/* mulberry32: small, seedable, identical in every JS engine. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Snake {
  constructor({ width = 12, height = 12, seed = 1 } = {}) {
    this.width = width;
    this.height = height;
    this.random = rng(seed);
    const cx = width >> 1, cy = height >> 1;
    this.body = [[cx, cy], [cx - 1, cy], [cx - 2, cy]]; /* head first */
    this.alive = true;
    this.score = 0;
    this.steps = 0;
    this.sinceFood = 0;
    this.placeFood();
  }

  key(x, y) { return y * this.width + x; }

  placeFood() {
    const taken = new Set(this.body.map(([x, y]) => this.key(x, y)));
    const free = [];
    for (let y = 0; y < this.height; ++y)
      for (let x = 0; x < this.width; ++x)
        if (!taken.has(this.key(x, y))) free.push([x, y]);
    this.food = free.length ? free[Math.floor(this.random() * free.length)] : null;
  }

  /* What moving in `move` would do, without doing it. */
  outcome(move) {
    const [hx, hy] = this.body[0];
    const [dx, dy] = DELTA[move];
    const nx = hx + dx, ny = hy + dy;
    if (nx < 0 || ny < 0 || nx >= this.width || ny >= this.height) {
      return { kind: "wall", reach: 0 };
    }
    const eats = this.food && nx === this.food[0] && ny === this.food[1];
    /* The tail vacates its cell this turn unless the snake grows. */
    const body = eats ? this.body : this.body.slice(0, -1);
    const occupied = new Set(body.map(([x, y]) => this.key(x, y)));
    if (occupied.has(this.key(nx, ny))) return { kind: "body", reach: 0 };

    /* Flood fill from the new head: a region smaller than the snake is a trap.
     * Crude -- the tail keeps retreating -- but it is what a careful player
     * checks, and it is cheap. */
    occupied.add(this.key(nx, ny));
    const length = body.length + 1;
    const seen = new Set([this.key(nx, ny)]);
    const stack = [[nx, ny]];
    while (stack.length && seen.size <= length) {
      const [x, y] = stack.pop();
      for (const [ddx, ddy] of Object.values(DELTA)) {
        const px = x + ddx, py = y + ddy;
        if (px < 0 || py < 0 || px >= this.width || py >= this.height) continue;
        const k = this.key(px, py);
        if (seen.has(k) || occupied.has(k)) continue;
        seen.add(k);
        stack.push([px, py]);
      }
    }
    const reach = seen.size;
    if (reach <= length && reach < this.width * this.height - length) {
      return { kind: "trap", reach };
    }
    if (eats) return { kind: "food", reach };
    const before = Math.abs(hx - this.food[0]) + Math.abs(hy - this.food[1]);
    const after = Math.abs(nx - this.food[0]) + Math.abs(ny - this.food[1]);
    return { kind: after < before ? "toward" : "away", reach };
  }

  outcomes() {
    return Object.fromEntries(MOVES.map((m) => [m, this.outcome(m)]));
  }

  /* The state text. `random` picks phrasing and sentence order; pass the
   * game's own generator when playing and a separate one when generating data
   * so the two streams do not interfere. */
  describe(outcomes = this.outcomes(), random = this.random) {
    const pick = (list) => list[Math.floor(random() * list.length)];
    const sentences = MOVES.map((m) => `${pick(LEADS)} ${m} ${pick(PHRASES[outcomes[m].kind])}.`);
    for (let i = sentences.length - 1; i > 0; --i) {
      const j = Math.floor(random() * (i + 1));
      [sentences[i], sentences[j]] = [sentences[j], sentences[i]];
    }
    return sentences.join(" ");
  }

  /* Best moves by the oracle's ranking: outcome first, then open space. */
  best(outcomes = this.outcomes()) {
    const score = (m) => RANK[outcomes[m].kind];
    const top = Math.min(...MOVES.map(score));
    return MOVES.filter((m) => score(m) === top);
  }

  step(move) {
    if (!this.alive) return;
    const { kind } = this.outcome(move);
    const [dx, dy] = DELTA[move];
    const [hx, hy] = this.body[0];
    ++this.steps;
    ++this.sinceFood;
    if (kind === "wall" || kind === "body") {
      this.alive = false;
      this.death = kind;
      return;
    }
    const eats = this.food && hx + dx === this.food[0] && hy + dy === this.food[1];
    this.body.unshift([hx + dx, hy + dy]);
    if (eats) {
      ++this.score;
      this.sinceFood = 0;
      this.placeFood();
      if (!this.food) this.alive = false, this.death = "won";
    } else {
      this.body.pop();
    }
  }
}
