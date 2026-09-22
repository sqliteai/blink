/* doom.mjs -- ViZDoom's Basic and Predict Position, the text Blink reads, and
 * the teacher that labels it.
 *
 * The two ViZDoom scenarios are played through a text interface: the model
 * sees the label boxes of the monsters on screen and picks one of four
 * actions. ViZDoom is a native
 * engine and its scenarios need ZDoom (ACS scripts, the label buffer), so it
 * cannot run in a browser. This module reimplements the two scenarios in
 * plain JavaScript so the whole demo -- data, headless evaluation and the
 * browser page -- runs on blink.wasm. One module serves all three, so the
 * text the model is trained on and the text it plays from are produced by
 * the same function.
 *
 * Every constant below was measured on ViZDoom 1.3.1 with its bundled
 * scenario files: positions, speeds and bounds from the game variables and
 * label positions, the projection from 2,000 label boxes. It is a faithful
 * model of the rules, not a bit-exact port: Doom's own random number table
 * and monster AI are replaced by a seeded generator and a statistical match.
 *
 * The rules:
 *
 *   - four actions: left, right, shoot, noop. Basic strafes, Predict Position
 *     turns. An action is held for `frameSkip` ticks, one tick at a time,
 *     stopping at the first tick that ends the episode;
 *   - success is a kill. Running out of time or decisions is a failure;
 *   - the model sees what is on screen: the box of the rendered monster in
 *     the last few observations, the view angle and the ammo.
 *
 * SPDX-License-Identifier: Apache-2.0 */

export const ACTIONS = ["left", "right", "shoot", "noop"];
export const SCENARIOS = ["basic", "predict_position"];
export const QUESTION = "Which action eliminates the monster before the deadline?";

/* The option text is what the model scores. It names the movement each
 * scenario actually uses, so the same slot means different buttons. */
export const OPTIONS = {
  basic: ["strafe left", "strafe right", "shoot", "wait"],
  predict_position: ["turn left", "turn right", "shoot", "wait"],
};

export const SCREEN_W = 320, SCREEN_H = 240;
export const FOCAL_X = 160;          /* 90-degree horizontal field of view */
export const FOCAL_Y = 192;          /* ZDoom's 1.2 pixel aspect */

/* Cacodemon: a 31-unit radius for collisions, a 60 x 62 sprite on screen. */
export const MONSTER_RADIUS = 31;
export const SPRITE_W = 60, SPRITE_H = 62;

export const SCENARIO = {
  basic: {
    /* A 448-unit-wide room. The player faces east and strafes along y; the
     * monster floats still against the far wall at a random y. */
    room: { x0: -448, x1: 64, y0: -192, y1: 256 },
    player: [-384, 32], playerY: [-176, 240],
    monsterX: [0, 0], monsterY: [-161, 222],
    spriteDrop: 11,                   /* sprite centre below eye level */
    ticks: 286,                       /* 300-tick timeout, 14 spent unholstering */
    ammo: 50,
  },
  predict_position: {
    /* A long hall. The player stands at the west end and only turns; the
     * monster wanders the east half. One rocket. */
    room: { x0: -608, x1: 256, y0: -640, y1: 640 },
    player: [-544, 0],
    monsterX: [0, 204], monsterY: [-586, 589],
    monsterBounds: { x0: -183, x1: 225, y0: -609, y1: 609 },
    spriteDrop: 8.3,
    ticks: 284,                       /* 300-tick timeout, 16 spent raising the launcher */
    ammo: 1,
  },
};

/* Strafe: every tick velocity = velocity * friction + thrust. Doom's
 * friction is 0xE800 / 0x10000; the fit to ViZDoom is exact. */
const STRAFE_THRUST = 0.75, FRICTION = 0.90625, STOP_SPEED = 0.0625;
/* Turn: Doom's slow turn for the first six held ticks, then full speed. */
const TURN_SLOW = 1.7578125, TURN_FAST = 3.515625, SLOW_TICKS = 6;
/* Pistol: the shot leaves on the fifth tick of a press; holding refires
 * every 14 ticks, and a refire is no longer accurate. */
const PISTOL_DELAY = 4, PISTOL_CYCLE = 14, PISTOL_SPREAD = 5.6;
/* Rocket: leaves eight ticks after the press, 20 units a tick, and hits when
 * its 11-unit box overlaps the monster's. */
const ROCKET_DELAY = 8, ROCKET_SPEED = 20, ROCKET_RADIUS = 11;
/* Monster: a step of 20 units every third tick in one of eight directions
 * (0 = east, counter-clockwise in 45-degree steps). The direction process is
 * the one measured on 400 ViZDoom episodes: the first heading, then, each
 * time the heading changes, how often it changes to each other one. It
 * zigzags diagonally (north-east <-> north-west) and runs long north-south,
 * which keeps its motion across the player's view steady while its depth
 * wobbles -- the property a rocket shot depends on. */
const MONSTER_STEP = 20, MONSTER_PERIOD = 3;
/* Run lengths: Doom's movecount, P_Random() & 15, plus a long tail fitted so
 * that a straight-line guess 36 ticks ahead (a rocket's flight) lands within
 * the monster as often as in ViZDoom: 55% across the view (ViZDoom 57%),
 * 44% in depth (43%), and the heading holds on 92% of steps (93%). */
const RUNS = { northSouth: 0.85, diagonal: 0.2, extra: 120 };
const FIRST_HEADING = [29, 61, 203, 4, 0, 2, 185, 23];
const TURNS = [
  [0, 12, 1, 98, 0, 114, 0, 4],
  [4, 0, 66, 286, 11, 0, 15, 4],
  [104, 11, 0, 38, 10, 1, 0, 97],
  [1, 324, 44, 0, 3, 2, 7, 0],
  [0, 26, 1, 4, 0, 3, 0, 18],
  [5, 0, 5, 8, 4, 0, 60, 388],
  [90, 10, 0, 0, 68, 26, 0, 38],
  [5, 5, 46, 0, 9, 393, 52, 0],
];

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

const RAD = Math.PI / 180;
const wrap = (deg) => ((deg % 360) + 360) % 360;
const between = (random, [lo, hi]) => lo + random() * (hi - lo);

export class Doom {
  constructor({ scenario = "basic", seed = 1, frameSkip = 4, maxSteps = 75, history = 4 } = {}) {
    if (!SCENARIOS.includes(scenario)) throw new Error(`unknown scenario ${scenario}`);
    this.scenario = scenario;
    this.spec = SCENARIO[scenario];
    this.frameSkip = frameSkip;
    this.maxSteps = maxSteps;
    this.historyLength = history;
    this.random = rng(seed * 2654435761);
    this.seed = seed;

    const s = this.spec;
    this.player = { x: s.player[0], y: s.player[1], angle: 0, vy: 0 };
    this.monster = {
      x: Math.round(between(this.random, s.monsterX)),
      y: Math.round(between(this.random, s.monsterY)),
      dir: -1, moves: 0, alive: true, dying: 0,
    };
    this.ammo = s.ammo;
    this.ticks = 0;
    this.steps = 0;
    this.turnHeld = 0;
    this.weapon = { busy: 0, fireAt: -1, refire: false, flash: 0 };
    this.rocket = null;
    this.explosion = null;
    this.shots = [];          /* recent hitscan puffs, for the renderer */
    this.done = false;
    this.success = false;
    this.outcome = "running";
    this.history = [];
    this.observe();
  }

  remainingTicks() {
    return Math.max(0, Math.min(this.spec.ticks - this.ticks,
      (this.maxSteps - this.steps) * this.frameSkip));
  }

  /* -- one Doom tick ------------------------------------------------------ */

  tick(action) {
    const p = this.player, s = this.spec;
    const left = action === "left", right = action === "right";
    const attack = action === "shoot";

    if (this.scenario === "basic") {
      p.vy = p.vy * FRICTION + (left ? STRAFE_THRUST : right ? -STRAFE_THRUST : 0);
      if (!left && !right && Math.abs(p.vy) < STOP_SPEED) p.vy = 0;
      p.y += p.vy;
      if (p.y < s.playerY[0] || p.y > s.playerY[1]) {
        p.y = Math.min(s.playerY[1], Math.max(s.playerY[0], p.y));
        p.vy = 0;
      }
    } else if (left || right) {
      const rate = this.turnHeld < SLOW_TICKS ? TURN_SLOW : TURN_FAST;
      p.angle = wrap(p.angle + (left ? rate : -rate));
      ++this.turnHeld;
    } else {
      this.turnHeld = 0;
    }

    this.weaponTick(attack);
    this.monsterTick();
    this.rocketTick();
    if (this.weapon.flash > 0) --this.weapon.flash;
    if (this.explosion && ++this.explosion.age > 12) this.explosion = null;
    this.shots = this.shots.filter((shot) => ++shot.age < 6);
    if (!this.monster.alive) ++this.monster.dying;
    ++this.ticks;
  }

  weaponTick(attack) {
    const w = this.weapon;
    if (w.busy === 0 && attack && this.ammo > 0) {
      const delay = this.scenario === "basic" ? PISTOL_DELAY : ROCKET_DELAY;
      w.busy = this.scenario === "basic" ? PISTOL_CYCLE : 40;
      w.fireAt = delay;
    }
    if (w.busy === 0) { w.refire = false; return; }
    if (w.fireAt === 0 && this.ammo > 0) this.fire();
    --w.fireAt;
    if (--w.busy === 0 && attack && this.ammo > 0 && this.scenario === "basic") {
      /* Still holding at the end of the cycle: refire, inaccurately. */
      w.refire = true;
      w.busy = PISTOL_CYCLE;
      w.fireAt = PISTOL_DELAY - 1;
    }
  }

  fire() {
    const p = this.player, m = this.monster;
    --this.ammo;
    this.weapon.flash = 4;
    if (this.scenario === "predict_position") {
      const a = p.angle * RAD;
      this.rocket = { x: p.x + 10 * Math.cos(a), y: p.y + 10 * Math.sin(a),
        dx: ROCKET_SPEED * Math.cos(a), dy: ROCKET_SPEED * Math.sin(a) };
      return;
    }
    /* Hitscan. The first shot of a press is dead on; a refire is spread. */
    let angle = p.angle;
    if (this.weapon.refire) angle += (this.random() - this.random()) * PISTOL_SPREAD;
    const a = angle * RAD;
    const fx = Math.cos(a), fy = Math.sin(a);
    const depth = (m.x - p.x) * fx + (m.y - p.y) * fy;
    const miss = Math.abs(-(m.x - p.x) * fy + (m.y - p.y) * fx);
    if (m.alive && depth > 0 && miss < MONSTER_RADIUS) {
      this.kill();
      this.shots.push({ x: m.x - 10 * fx, y: m.y - 10 * fy, age: 0, blood: true });
    } else {
      const reach = (this.spec.room.x1 - 1 - p.x) / fx;   /* to the far wall */
      this.shots.push({ x: p.x + fx * reach, y: p.y + fy * reach, age: 0, blood: false });
    }
  }

  monsterTick() {
    const m = this.monster, b = this.spec.monsterBounds;
    if (!b || !m.alive || this.ticks % MONSTER_PERIOD !== 2) return;
    if (m.dir < 0 || m.moves <= 0) this.newDirection();
    const a = m.dir * 45 * RAD;
    let nx = m.x + Math.round(MONSTER_STEP * Math.cos(a));
    let ny = m.y + Math.round(MONSTER_STEP * Math.sin(a));
    if (nx < b.x0 || nx > b.x1 || ny < b.y0 || ny > b.y1) {
      this.newDirection();                         /* blocked: pick a heading that fits */
      const c = m.dir * 45 * RAD;
      nx = m.x + Math.round(MONSTER_STEP * Math.cos(c));
      ny = m.y + Math.round(MONSTER_STEP * Math.sin(c));
      if (nx < b.x0 || nx > b.x1 || ny < b.y0 || ny > b.y1) return;
    }
    m.x = nx; m.y = ny; --m.moves;
  }

  newDirection() {
    const m = this.monster, b = this.spec.monsterBounds;
    const fits = (d) => {
      const a = d * 45 * RAD;
      const nx = m.x + Math.round(MONSTER_STEP * Math.cos(a)), ny = m.y + Math.round(MONSTER_STEP * Math.sin(a));
      return nx >= b.x0 && nx <= b.x1 && ny >= b.y0 && ny <= b.y1;
    };
    let weights = (m.dir < 0 ? FIRST_HEADING : TURNS[m.dir]).map((w, d) => (fits(d) ? w : 0));
    if (!weights.some((w) => w)) weights = weights.map((_, d) => (fits(d) && d !== m.dir ? 1 : 0));
    if (!weights.some((w) => w)) return;
    let r = this.random() * weights.reduce((sum, w) => sum + w, 0);
    let d = 0;
    while (r >= weights[d]) r -= weights[d++];
    m.dir = d;
    const long = this.random() < ((d === 2 || d === 6) ? RUNS.northSouth : d % 2 ? RUNS.diagonal : 0);
    m.moves = long ? 17 + Math.floor(this.random() * RUNS.extra) : 1 + Math.floor(this.random() * 16);
  }

  rocketTick(live = true) {
    const r = this.rocket, m = this.monster, room = this.spec.room;
    if (!r) return;
    r.x += r.dx;
    r.y += r.dy;
    const reach = MONSTER_RADIUS + ROCKET_RADIUS;
    if (m.alive && Math.abs(r.x - m.x) < reach && Math.abs(r.y - m.y) < reach) {
      if (live) this.kill();      /* after the deadline it only bursts */
    } else if (r.x > room.x1 - 20 || r.x < room.x0 + ROCKET_RADIUS ||
               Math.abs(r.y) > room.y1 - ROCKET_RADIUS) {
      /* Missed: it bursts on a wall. */
    } else {
      return;
    }
    this.explosion = { x: r.x, y: r.y, age: 0 };
    this.rocket = null;
  }

  kill() {
    this.monster.alive = false;
    this.success = true;
  }

  /* -- one decision -------------------------------------------------------- */

  step(action) {
    for (const _ of this.stepTicks(action));   // eslint-disable-line no-unused-vars
  }

  /* The same decision one tick at a time, for the browser to animate. */
  *stepTicks(action) {
    if (this.done) throw new Error("episode is over");
    const hold = Math.min(this.frameSkip, this.remainingTicks());
    for (let i = 0; i < hold && !this.success; ++i) {
      this.tick(action);
      yield;
    }
    ++this.steps;
    if (this.success) this.outcome = "killed";
    else if (this.ticks >= this.spec.ticks) this.outcome = "timeout";
    else if (this.steps >= this.maxSteps) this.outcome = "deadline";
    this.done = this.outcome !== "running";
    this.observe();
  }

  /* Advance the animations only (a dying monster, a rocket still in the
   * air, an explosion) once the episode is over. Changes no outcome. */
  idle() {
    const m = this.monster;
    if (!m.alive) ++m.dying;
    if (this.weapon.flash > 0) --this.weapon.flash;
    if (this.explosion && ++this.explosion.age > 12) this.explosion = null;
    this.shots = this.shots.filter((shot) => ++shot.age < 6);
    this.rocketTick(false);
  }

  /* -- what is on screen --------------------------------------------------- */

  /* The monster's label box as ViZDoom reports it, clipped to the screen,
   * or null when no pixel of it is visible. */
  project(x = this.monster.x, y = this.monster.y) {
    const p = this.player, a = p.angle * RAD;
    const dx = x - p.x, dy = y - p.y;
    const depth = dx * Math.cos(a) + dy * Math.sin(a);
    const lateral = dx * Math.sin(a) - dy * Math.cos(a);
    if (depth < 16) return null;
    const cx = SCREEN_W / 2 + FOCAL_X * lateral / depth;
    const cy = SCREEN_H / 2 + FOCAL_Y * this.spec.spriteDrop / depth;
    const w = FOCAL_X * SPRITE_W / depth, h = FOCAL_Y * SPRITE_H / depth;
    const x0 = Math.max(0, Math.round(cx - w / 2)), x1 = Math.min(SCREEN_W, Math.round(cx + w / 2));
    const y0 = Math.max(0, Math.round(cy - h / 2)), y1 = Math.min(SCREEN_H, Math.round(cy + h / 2));
    if (x1 <= x0 || y1 <= y0) return null;
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0, depth, lateral, sx: cx, sy: cy, sw: w, sh: h };
  }

  observe() {
    const box = this.monster.alive ? this.project() : null;
    const frame = {
      tick: this.ticks,
      ammo: this.ammo,
      angle: this.player.angle,
      target: box && {
        cx: Math.floor(box.x + box.w / 2), cy: Math.floor(box.y + box.h / 2), w: box.w, h: box.h,
        world: [this.monster.x, this.monster.y],   /* privileged: teacher only */
      },
      player: [this.player.x, this.player.y],
    };
    this.history.push(frame);
    if (this.history.length > this.historyLength) this.history.shift();
  }

  /* The state text: one clause per remembered observation, oldest first,
   * each with the view angle so a shift on screen can be told apart from a
   * turn. A box is written as its centre and size, which is the same
   * information as a corner and a size with the one addition the model would
   * otherwise have to learn to do in decimal. */
  describe() {
    const now = this.history[this.history.length - 1];
    const parts = [`${this.scenario.replace("_", " ")}. ammo ${now.ammo}. ${this.remainingTicks()} ticks left.`];
    for (const f of this.history) {
      const age = now.tick - f.tick;
      const seen = f.target ? `x ${f.target.cx} y ${f.target.cy} w ${f.target.w} h ${f.target.h}` : "no monster";
      parts.push(`${age ? `t-${age}` : "now"}: a ${Math.round(f.angle) % 360} ${seen}.`);
    }
    return parts.join(" ");
  }

  /* -- the teacher ---------------------------------------------------------- */

  teacher() {
    return this.scenario === "basic" ? this.basicTeacher() : this.predictTeacher();
  }

  /* Visible-only: shoot when the monster is centred, otherwise strafe toward
   * it, allowing for the drift left over from the last strafe. Strafing left
   * slides the monster right on screen. */
  basicTeacher() {
    const h = this.history, now = h[h.length - 1];
    if (!now.target) return "noop";
    const offset = now.target.cx - SCREEN_W / 2;
    const before = h.length > 1 ? h[h.length - 2].target : null;
    const drift = before ? now.target.cx - before.cx : 0;
    const predicted = offset + drift;
    const tolerance = Math.max(3, now.target.w >> 2);
    if (Math.abs(offset) <= tolerance && Math.abs(predicted) <= tolerance + 2) return "shoot";
    if (Math.abs(predicted) <= tolerance) return "noop";
    return predicted > 0 ? "right" : "left";
  }

  /* Privileged: reads the monster's world position (which the model never
   * sees), estimates its velocity from the remembered observations, solves
   * for where the rocket meets it, and turns toward that point. */
  predictTeacher() {
    const h = this.history, now = h[h.length - 1];
    if (now.ammo <= 0 || this.weapon.busy || !now.target) return "noop";
    const [px, py] = now.player, [mx, my] = now.target.world;
    let vx = 0, vy = 0;
    const old = h.find((f) => f !== now && f.target && f.tick < now.tick);
    if (old) {
      const dt = now.tick - old.tick;
      vx = (mx - old.target.world[0]) / dt;
      vy = (my - old.target.world[1]) / dt;
    }
    let t = 0, dx = mx - px, dy = my - py;
    for (let i = 0; i < 8; ++i) {
      dx = mx + vx * t - px;
      dy = my + vy * t - py;
      t = ROCKET_DELAY + Math.hypot(dx, dy) / ROCKET_SPEED;
    }
    const aim = Math.atan2(dy, dx) / RAD;
    const diff = ((aim - now.angle + 540) % 360) - 180;       /* positive: turn left */
    if (Math.abs(diff) <= 1.5) return "shoot";
    const rate = this.turnHeld >= SLOW_TICKS ? TURN_FAST : TURN_SLOW;
    if (Math.abs(diff) < rate * this.frameSkip / 2) return "noop";
    return diff > 0 ? "left" : "right";
  }
}
