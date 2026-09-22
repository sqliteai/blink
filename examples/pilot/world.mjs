/* world.mjs -- the road, the traffic, the planner and the oracle that labels it.
 *
 * One module serves every side, as in examples/snake: make_data.mjs uses it
 * to write the training corpus, drive.mjs runs headless trips against the
 * model, and index.html renders the same world in the browser. The text the
 * model is trained on and the text it drives from come from one function.
 *
 * The trip: start in Millbrook, stop at a sign and a light, take the on-ramp,
 * merge onto Interstate 08, move over for the exit and park in Cedar Town.
 * Positions are route
 * coordinates: `s` metres along the route and `d` metres to the left of the
 * rightmost interstate lane.
 *
 * Every decision works the same way: geometry, traffic prediction and
 * control stay local. Each of six maneuvers is rolled out three seconds
 * ahead against predicted traffic, signals and lane geometry, and the planner
 * writes one sentence per maneuver saying what it would lead to ("left cuts
 * off a car.", "keep runs the red light."). The model scores the six
 * maneuvers against that text. Nothing in the text gives the ranking.
 *
 * SPDX-License-Identifier: Apache-2.0 */

export const OPTIONS = ["faster", "keep", "slower", "stop", "left", "right"];
export const QUESTION = "Which maneuver is safe and legal and makes the most progress?";

export const LANE_W = 3.6;
export const HORIZON = 3.0;          /* seconds each maneuver is rolled out */
export const DECISION_S = 0.25;      /* seconds between decisions           */
const PRED_DT = 0.1;
export const EGO = { L: 4.7, W: 1.95 };

const kmh = (k) => k / 3.6;

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

/* ---------------------------------------------------------------- route -- */

/* Turns are in degrees, positive to the right. Screen coordinates: x east,
 * y south, heading 0 = east, so a right turn increases the heading. */
const SEGMENTS = [
  { zone: "town", straight: 200, limit: kmh(50) },
  { zone: "town", arc: 22, turn: 90, limit: kmh(28) },
  { zone: "town", straight: 260, limit: kmh(50) },
  { zone: "ramp", arc: 60, turn: 90, limit: kmh(45) },
  { zone: "ramp", straight: 140, limit: kmh(90) },
  { zone: "merge", straight: 260, limit: kmh(105) },
  { zone: "highway", straight: 500, limit: kmh(105) },
  { zone: "highway", arc: 800, turn: -18, limit: kmh(105) },
  { zone: "highway", straight: 600, limit: kmh(105) },
  { zone: "highway", arc: 800, turn: 18, limit: kmh(105) },
  { zone: "highway", straight: 350, limit: kmh(105) },
  { zone: "exit", straight: 260, limit: kmh(105) },
  { zone: "exitramp", arc: 90, turn: 50, limit: kmh(60) },
  { zone: "exitramp", straight: 120, limit: kmh(60) },
  { zone: "exitramp", arc: 25, turn: 70, limit: kmh(30) },
  { zone: "cedar", straight: 280, limit: kmh(50) },
];

/* Lanes that exist in each zone. -1 is the town road, the ramps and the
 * interstate's merge and exit lanes; 0..2 are the interstate. */
const ZONE_LANES = {
  town: { "-1": "drive", 0: "oncoming" },
  ramp: { "-1": "drive" },
  merge: { "-1": "drive", 0: "drive", 1: "drive", 2: "drive" },
  highway: { 0: "drive", 1: "drive", 2: "drive" },
  exit: { "-1": "drive", 0: "drive", 1: "drive", 2: "drive" },
  exitramp: { "-1": "drive" },
  cedar: { "-1": "drive", 0: "oncoming" },
};

export const ZONE_NAMES = {
  town: "Millbrook", ramp: "On-ramp", merge: "Merge", highway: "Interstate 08",
  exit: "Exit 14", exitramp: "Exit ramp", cedar: "Cedar Town",
};

function buildRoute() {
  const xs = [], ys = [], hs = [], zones = [], limits = [];
  const zoneRanges = [];
  let x = 0, y = 0, h = -Math.PI / 2;
  const push = (seg) => {
    xs.push(x); ys.push(y); hs.push(h); zones.push(seg.zone); limits.push(seg.limit);
  };
  for (const seg of SEGMENTS) {
    const start = xs.length;
    if (seg.straight) {
      for (let i = 0; i < seg.straight; ++i) {
        push(seg);
        x += Math.cos(h); y += Math.sin(h);
      }
    } else {
      const length = Math.round(seg.arc * Math.abs(seg.turn) * Math.PI / 180);
      const dh = (seg.turn * Math.PI / 180) / length;
      for (let i = 0; i < length; ++i) {
        push(seg);
        x += Math.cos(h + dh / 2); y += Math.sin(h + dh / 2);
        h += dh;
      }
    }
    const last = zoneRanges[zoneRanges.length - 1];
    if (last && last.zone === seg.zone) last.end = xs.length;
    else zoneRanges.push({ zone: seg.zone, start, end: xs.length });
  }
  push(SEGMENTS[SEGMENTS.length - 1]);
  return { xs, ys, hs, zones, limits, zoneRanges, length: xs.length - 1 };
}

export const ROUTE = buildRoute();
const zoneStart = (z) => ROUTE.zoneRanges.find((r) => r.zone === z).start;
const zoneEnd = (z) => ROUTE.zoneRanges.find((r) => r.zone === z).end;

export const MARKS = {
  mergeStart: zoneStart("merge"),
  mergeEnd: zoneEnd("merge"),
  exitStart: zoneStart("exit"),
  exitEnd: zoneEnd("exit"),
  townEnd: zoneEnd("town"),
  cedarStart: zoneStart("cedar"),
  destination: ROUTE.length - 30,
};
/* The interstate runs past both ends of the part the route uses. */
MARKS.hwStart = MARKS.mergeStart - 600;
MARKS.hwEnd = MARKS.exitEnd + 600;

export const SIGNS = [
  { id: 0, s: 120 },
  { id: 1, s: MARKS.cedarStart + 110 },
];
export const LIGHTS = [
  { id: 0, s: 200 + 35 + 170, green: 10, amber: 3, red: 9 },
];

export function zoneAt(s) {
  return ROUTE.zones[Math.max(0, Math.min(ROUTE.length, Math.floor(s)))];
}
export function limitAt(s) {
  return ROUTE.limits[Math.max(0, Math.min(ROUTE.length, Math.floor(s)))];
}
export function laneKind(s, lane) {
  if (s < 0 || s > ROUTE.length) return null;
  return ZONE_LANES[zoneAt(s)][lane] ?? null;
}
export function lanesAt(s) {
  return Object.keys(ZONE_LANES[zoneAt(s)]).map(Number).sort((a, b) => a - b);
}

/* Pose on the route: world x, y and heading, `d` metres left of the route. */
export function pose(s, d = 0) {
  const n = ROUTE.length;
  const c = Math.max(0, Math.min(n - 1e-6, s));
  const i = Math.floor(c), f = c - i;
  let h0 = ROUTE.hs[i], h1 = ROUTE.hs[i + 1];
  const h = h0 + (h1 - h0) * f;
  let x = ROUTE.xs[i] + (ROUTE.xs[i + 1] - ROUTE.xs[i]) * f;
  let y = ROUTE.ys[i] + (ROUTE.ys[i + 1] - ROUTE.ys[i]) * f;
  const over = s - c;                 /* past either end: continue straight */
  x += Math.cos(h) * over; y += Math.sin(h) * over;
  return { x: x + Math.sin(h) * d, y: y - Math.cos(h) * d, h };
}

/* Pose on the interstate, which leaves the route before the merge and after
 * the exit and carries on straight. */
export function highwayPose(s, d) {
  const anchor = s < MARKS.mergeStart ? MARKS.mergeStart : s > MARKS.exitEnd ? MARKS.exitEnd : null;
  if (anchor === null) return pose(s, d);
  const p = pose(anchor, d);
  const over = s - anchor;
  return { x: p.x + Math.cos(p.h) * over, y: p.y + Math.sin(p.h) * over, h: p.h };
}

export function lightState(light, t) {
  const cycle = light.green + light.amber + light.red;
  const phase = (((t + light.offset) % cycle) + cycle) % cycle;
  return phase < light.green ? "green" : phase < light.green + light.amber ? "amber" : "red";
}

/* The lanes a vehicle at lateral position d overlaps. */
function lanesOverlapped(d, width = EGO.W) {
  const lanes = [];
  for (let l = -1; l <= 2; ++l) {
    if (Math.abs(l * LANE_W - d) < LANE_W / 2 + width / 2 - 0.35) lanes.push(l);
  }
  return lanes;
}

/* Where the route wants the car, as a lane range, or null when any lane is
 * fine. Off the merge lane on the merge, into lane 0 ahead of the exit and
 * into the exit lane on the exit. */
function desiredLanes(s) {
  const z = zoneAt(s);
  if (z === "merge") return [0, 2];
  if (z === "exit") return [-1, -1];
  if (z === "highway" && s > MARKS.exitStart - 1200) return [0, 0];
  return null;
}
function laneDistance(lane, range) {
  return lane < range[0] ? range[0] - lane : lane > range[1] ? lane - range[1] : 0;
}

/* ----------------------------------------------------------- traffic -- */

const IDM = { a: 1.4, b: 2.5, T: 1.2, s0: 2.5 };

/* A traffic car interacts with the ego car only where they share geometry:
 * interstate cars only on the stretch the route uses. */
function interacts(car) {
  if (car.group === "oncoming") return false;
  if (car.group === "hw") return car.s > MARKS.mergeStart - 3 && car.s < MARKS.exitEnd + 3;
  return true;
}

function overlap(a, aL, aW, b) {
  return Math.abs(a.s - b.s) < (aL + b.L) / 2 && Math.abs(a.d - b.d) < (aW + b.W) / 2;
}

function idm(v, v0, gap, dv) {
  const star = IDM.s0 + Math.max(0, v * IDM.T + (v * dv) / (2 * Math.sqrt(IDM.a * IDM.b)));
  const free = 1 - Math.pow(v / Math.max(v0, 0.1), 4);
  const inter = gap < 1e6 ? (star / Math.max(gap, 0.1)) ** 2 : 0;
  return Math.max(-9, Math.min(IDM.a, IDM.a * (free - inter)));
}

/* One traffic step. `ego` is the ego car (real or predicted): traffic behind
 * it treats it as a leader, so a cut-in makes the follower brake. */
function stepTraffic(cars, ego, t, dt) {
  const accel = new Array(cars.length).fill(0);
  for (let i = 0; i < cars.length; ++i) {
    const c = cars[i];
    if (c.group === "oncoming") continue;
    let gap = 1e9, lead = 0;
    const front = c.s + c.L / 2;
    for (const o of cars) {
      if (o === c || o.group === "oncoming" || o.lane !== c.lane) continue;
      if ((o.group === "hw") !== (c.group === "hw")) continue;
      if (o.s <= c.s) continue;
      const g = o.s - o.L / 2 - front;
      if (g < gap) { gap = g; lead = o.v; }
    }
    if (ego && interacts(c) && ego.s > c.s &&
        Math.abs(ego.d - c.d) < (EGO.W + c.W) / 2 + 0.2) {
      const g = ego.s - EGO.L / 2 - front;
      if (g < gap) { gap = g; lead = ego.v; }
    }
    if (c.group !== "hw") {
      for (const light of LIGHTS) {
        const g = light.s - front - 1;
        if (g < -0.5 || g > 120) continue;
        const state = lightState(light, t);
        if (state === "red" || (state === "amber" && g > (c.v * c.v) / (2 * 3))) {
          if (g < gap) { gap = Math.max(g, 0.05); lead = 0; }
        }
      }
      for (const sign of SIGNS) {
        if (c.signs.has(sign.id)) continue;
        const g = sign.s - front - 1;
        if (g < -0.5 || g > 120) continue;
        if (g < gap) { gap = Math.max(g, 0.05); lead = 0; }
      }
    }
    let v0 = c.v0;
    for (let k = 0; k <= 40; k += 10) v0 = Math.min(v0, limitAt(c.s + k) * c.pace);
    accel[i] = idm(c.v, v0, gap, c.v - lead);
  }
  for (let i = 0; i < cars.length; ++i) {
    const c = cars[i];
    if (c.group === "oncoming") { c.s -= c.v * dt; continue; }
    c.v = Math.max(0, c.v + accel[i] * dt);
    c.s += c.v * dt;
    c.a = accel[i];
    if (c.group !== "hw") {
      for (const sign of SIGNS) {
        if (c.signs.has(sign.id)) continue;
        const g = sign.s - (c.s + c.L / 2);
        if (g < 3.5 && g > -1 && c.v < 0.3) {
          c.wait = (c.wait ?? 0) + dt;
          if (c.wait > 1) { c.signs.add(sign.id); c.wait = 0; }
        }
      }
    }
  }
}

const PALETTE = ["#d94f45", "#3f6fd8", "#e2e2dc", "#2c2f36", "#8c9197", "#e0a93b", "#3c9a6b", "#7a4fc4"];

function makeCar(random, group, s, lane, v0, truck = false) {
  return {
    id: 0, group, s, lane, d: lane * LANE_W, v: v0, v0, a: 0,
    L: truck ? 12 : 4.4 + random() * 0.5, W: truck ? 2.5 : 1.85,
    truck, color: truck ? "#c9ccd1" : PALETTE[Math.floor(random() * PALETTE.length)],
    pace: 0.92 + random() * 0.1, signs: new Set(),
  };
}

function spawnTraffic(random, density) {
  const cars = [];
  /* Town traffic ahead of the start, queueing at the same sign and light. */
  for (const [s, v] of [[62, 0], [180, 8], [330, 11]]) {
    if (random() < 0.8 * density + 0.2) {
      cars.push(makeCar(random, "town", s + random() * 20, -1, v));
      cars[cars.length - 1].v0 = kmh(46 + random() * 6);
    }
  }
  if (random() < density) {
    const c = makeCar(random, "cedar", MARKS.cedarStart + 30 + random() * 40, -1, kmh(30));
    c.v0 = kmh(28 + random() * 8);
    cars.push(c);
  }
  /* Oncoming traffic, for the picture only. */
  for (const [a, b] of [[0, MARKS.townEnd], [MARKS.cedarStart, ROUTE.length]]) {
    for (let k = 0; k < 3; ++k) {
      const c = makeCar(random, "oncoming", a + random() * (b - a), 0, kmh(40 + random() * 10));
      c.range = [a, b];
      cars.push(c);
    }
  }
  /* The interstate: trucks and slow cars on the right, fast on the left. */
  const ring = MARKS.hwEnd - MARKS.hwStart;
  const lanes = [
    { lane: 0, spacing: 150, v: [20, 24], trucks: 0.5 },
    { lane: 1, spacing: 190, v: [24.5, 27.5], trucks: 0.15 },
    { lane: 2, spacing: 240, v: [29, 33], trucks: 0 },
  ];
  for (const spec of lanes) {
    const count = Math.round((ring / spec.spacing) * density);
    const step = ring / Math.max(count, 1);
    for (let k = 0; k < count; ++k) {
      const s = MARKS.hwStart + k * step + random() * step * 0.5;
      const v0 = spec.v[0] + random() * (spec.v[1] - spec.v[0]);
      const truck = random() < spec.trucks;
      const c = makeCar(random, "hw", s, spec.lane, truck ? Math.min(v0, 22) : v0, truck);
      c.pace = 1;
      cars.push(c);
    }
  }
  cars.forEach((c, i) => (c.id = i + 1));
  return cars;
}

/* ----------------------------------------------------------- ego car -- */

function stopPoints(ego, t, lane) {
  const front = ego.s + EGO.L / 2;
  const points = [];
  for (const light of LIGHTS) {
    const state = lightState(light, t);
    if (light.s > front - 0.5 && state !== "green") points.push({ s: light.s, why: "line" });
  }
  for (const sign of SIGNS) {
    if (!ego.signs.has(sign.id) && sign.s > front - 0.5) points.push({ s: sign.s, why: "line" });
  }
  if (MARKS.destination > front - 0.5) points.push({ s: MARKS.destination, why: "dest" });
  const end = laneEnd(ego.s, lane);
  if (end !== null) points.push({ s: end, why: "line" });
  return points;
}

/* First point ahead (within 300 m) where `lane` stops being drivable. */
function laneEnd(s, lane) {
  if (laneKind(s, lane) !== "drive") return null;
  for (let k = 2; k <= 300; k += 2) {
    if (laneKind(s + k, lane) !== "drive") {
      let lo = s + k - 2, hi = s + k;
      while (hi - lo > 0.1) {
        const mid = (lo + hi) / 2;
        if (laneKind(mid, lane) === "drive") lo = mid; else hi = mid;
      }
      return lo;
    }
  }
  return null;
}

function leadOf(ego, cars, d) {
  let best = null, gap = 1e9;
  for (const c of cars) {
    if (!interacts(c) || c.s <= ego.s) continue;
    if (Math.abs(c.d - d) >= (EGO.W + c.W) / 2) continue;
    const g = c.s - c.L / 2 - (ego.s + EGO.L / 2);
    if (g < gap) { gap = g; best = c; }
  }
  return best ? { car: best, gap } : null;
}

/* Longitudinal command for a maneuver. `stop` approaches the nearest line,
 * lane end or queue and stops there, creeping up if it stopped short. */
function accelFor(option, ego, cars, t) {
  const limit = limitAt(ego.s);
  switch (option) {
    case "faster": return Math.max(0, Math.min(2.2, (limit - ego.v) / 0.8));
    case "slower": return ego.v > 0 ? -2.5 : 0;
    case "stop": {
      const front = ego.s + EGO.L / 2;
      let dist = Infinity;
      for (const p of stopPoints(ego, t, ego.lane)) dist = Math.min(dist, p.s - front - 0.5);
      for (const d of new Set([ego.d, ego.lane * LANE_W])) {
        const lead = leadOf(ego, cars, d);
        if (lead) dist = Math.min(dist, lead.gap - 3);
      }
      if (dist > 200) return ego.v > 0 ? -5 : 0;
      if (dist <= 0.05) return -8;
      /* Brake to stop exactly at the point, or creep up to it. */
      const need = (ego.v * ego.v) / (2 * dist);
      if (need > 1.2) return -Math.min(8, need);
      return Math.min(1.0, (Math.sqrt(2 * 1.0 * dist) * 0.7 - ego.v) * 1.5);
    }
    default: return 0;
  }
}

function stepEgo(ego, option, cars, t, dt) {
  const a = accelFor(option, ego, cars, t);
  ego.a = ego.v > 0 || a > 0 ? a : 0;
  ego.v = Math.max(0, ego.v + a * dt);
  ego.s += ego.v * dt;
  const err = ego.lane * LANE_W - ego.d;
  ego.dv = Math.max(-1.6, Math.min(1.6, err * 1.6));
  ego.d += ego.dv * dt;
  const front = ego.s + EGO.L / 2;
  for (const sign of SIGNS) {
    if (ego.signs.has(sign.id)) continue;
    const g = sign.s - front;
    if (g < 4 && g > -1 && ego.v < 0.3) {
      ego.signWait += dt;
      if (ego.signWait >= 1) { ego.signs.add(sign.id); ego.signWait = 0; }
    }
  }
}

function cloneEgo(ego) {
  return { ...ego, signs: new Set(ego.signs) };
}
function cloneCar(c) {
  return { ...c, signs: new Set(c.signs) };
}

/* ----------------------------------------------------------- planner -- */

/* Outcomes, best first. The oracle picks the lowest rank. */
export const RANKS = {
  best: 0, good: 1, slow: 2, idle: 3, away: 4, close: 5, speeding: 6,
  illegal: 7, cutoff: 8, crash: 9, nolane: 10, oncoming: 10,
};

const PHRASES = {
  best: ["makes the most progress", "is the quickest safe choice", "gets there fastest"],
  wait: ["waits safely", "holds safely"],
  stopline: ["stops at the line", "halts at the line"],
  stopcar: ["stops behind the car", "queues behind the car"],
  park: ["parks at the destination", "arrives and parks"],
  good: ["is safe but a bit slower", "is fine but slower"],
  slow: ["is safe but much slower", "crawls along"],
  idle: ["stops for no reason", "stands still for nothing"],
  away: ["heads away from the route lane", "drifts off the route lane"],
  close: ["follows too closely", "tailgates the car ahead"],
  speeding: ["is too fast for the road ahead", "breaks the speed limit"],
  red: ["runs the red light", "goes through on red"],
  sign: ["skips the stop sign", "rolls the stop sign"],
  dest: ["drives past the destination", "overshoots the destination"],
  laneend: ["runs out of lane", "hits the end of the lane"],
  exit: ["misses the exit", "stays on past the exit"],
  cutoff: ["cuts off a car", "cuts in front of a car"],
  crash: ["hits a car", "crashes into a car"],
  nolane: ["has no lane there", "drives off the road"],
  oncoming: ["goes into oncoming traffic", "crosses into oncoming cars"],
};

function worse(current, kind, why) {
  if (!current || RANKS[kind] > RANKS[current.kind]) return { kind, why: why ?? kind };
  return current;
}

/* Roll one maneuver out HORIZON seconds against predicted traffic. */
function rollout(world, option, near) {
  const ego = cloneEgo(world.ego);
  const cars = near.map(cloneCar);
  const lateral = option === "left" ? 1 : option === "right" ? -1 : 0;
  const path = [[ego.s, ego.d]];
  const startLane = ego.lane;
  if (lateral) {
    const target = ego.lane + lateral;
    const kind = laneKind(ego.s + 3, target);
    if (kind !== "drive") {
      const k = kind === "oncoming" ? "oncoming" : "nolane";
      const reach = Math.max(8, ego.v * 1.5);
      path.push([ego.s + reach, ego.d + lateral * LANE_W]);
      return { option, kind: k, why: k, progress: 0, moved: 0, endV: ego.v, path };
    }
    ego.lane = target;
  }
  let bad = null;
  let t = world.t;
  const steps = Math.round(HORIZON / PRED_DT);
  for (let i = 0; i < steps; ++i) {
    t += PRED_DT;
    const before = ego.s + EGO.L / 2;
    stepTraffic(cars, ego, t, PRED_DT);
    stepEgo(ego, option, cars, t, PRED_DT);
    const front = ego.s + EGO.L / 2;
    for (const l of lanesOverlapped(ego.d)) {
      const kind = laneKind(front, l);
      if (kind === "drive") continue;
      if (kind === "oncoming") bad = worse(bad, "oncoming");
      else if (lateral && l === ego.lane) bad = worse(bad, "nolane");
      else bad = worse(bad, "illegal", l >= 0 && front > MARKS.exitEnd - 5 ? "exit" : "laneend");
    }
    for (const sign of SIGNS) {
      if (!ego.signs.has(sign.id) && before < sign.s && front >= sign.s) bad = worse(bad, "illegal", "sign");
    }
    for (const light of LIGHTS) {
      if (before < light.s && front >= light.s && lightState(light, t) === "red") bad = worse(bad, "illegal", "red");
    }
    if (before < MARKS.destination && front >= MARKS.destination) bad = worse(bad, "illegal", "dest");
    for (const c of cars) {
      if (interacts(c) && overlap(ego, EGO.L, EGO.W, c)) bad = worse(bad, "crash");
    }
    if (ego.v > limitAt(ego.s) + 1.5) bad = worse(bad, "speeding");
    if ((i + 1) % 3 === 0) path.push([ego.s, ego.d]);
  }

  /* Terminal checks: what the end state commits the car to. */
  const front = ego.s + EGO.L / 2;
  const lead = leadOf(ego, cars, ego.lane * LANE_W) ?? leadOf(ego, cars, ego.d);
  if (lead) {
    const need = ego.v > lead.car.v ? (ego.v ** 2 - lead.car.v ** 2) / (2 * 7) : 0;
    if (lead.gap - need < 1) bad = worse(bad, "crash");
    else if (ego.v > 1 && lead.gap < 0.6 * ego.v + 3) bad = worse(bad, "close");
  }
  if (lateral) {
    let follower = null, gap = 1e9;
    for (const c of cars) {
      if (!interacts(c) || c.s >= ego.s) continue;
      if (Math.abs(c.d - ego.lane * LANE_W) >= (EGO.W + c.W) / 2) continue;
      const g = ego.s - EGO.L / 2 - (c.s + c.L / 2);
      if (g < gap) { gap = g; follower = c; }
    }
    if (follower && (gap < 2 ||
        (follower.v > ego.v && (follower.v ** 2 - ego.v ** 2) / (2 * (gap - 2)) > 3.5))) {
      bad = worse(bad, "cutoff");
    }
  }
  const brake = (ego.v * ego.v) / (2 * 7);
  for (const light of LIGHTS) {
    const g = light.s - front;
    if (g > 0 && lightState(light, t) !== "green" && brake > g + 0.5) bad = worse(bad, "illegal", "red");
  }
  for (const sign of SIGNS) {
    const g = sign.s - front;
    if (!ego.signs.has(sign.id) && g > 0 && brake > g + 0.5) bad = worse(bad, "illegal", "sign");
  }
  const toDest = MARKS.destination - front;
  if (toDest > 0 && brake > toDest + 0.5) bad = worse(bad, "illegal", "dest");
  const end = laneEnd(ego.s, ego.lane);
  if (end !== null && brake > end - front + 0.5) {
    bad = worse(bad, "illegal", ego.lane >= 0 && end > MARKS.exitEnd - 5 ? "exit" : "laneend");
  }
  const reach = (ego.v * ego.v) / (2 * 2.5);
  for (let k = 5; k <= reach; k += 5) {
    const limit = limitAt(ego.s + k);
    if (ego.v > limit + 1.5 && (ego.v ** 2 - limit ** 2) / (2 * k) > 2.5) {
      bad = worse(bad, "speeding");
      break;
    }
  }

  const moved = ego.s - world.ego.s;
  let progress = moved + ego.v * 1.0;
  let toward = 0;
  const want = desiredLanes(world.ego.s);
  if (lateral) {
    progress -= 4;
    if (want) toward = Math.sign(laneDistance(startLane, want) - laneDistance(ego.lane, want));
    if (toward > 0) progress += 25;
  }
  if (!bad && toward < 0) bad = { kind: "away", why: "away" };

  let stopWhy = null;
  if (option === "stop") {
    const points = stopPoints(world.ego, world.t, world.ego.lane);
    const lineAhead = points.some((p) => p.s - front < 60);
    const carAhead = lead && lead.gap < 25;
    if (MARKS.destination - front < 40) stopWhy = "park";
    else if (lineAhead) stopWhy = "stopline";
    else if (carAhead) stopWhy = "stopcar";
  }
  return {
    option, kind: bad?.kind ?? null, why: bad?.why ?? null, progress, moved,
    endV: ego.v, path, stopWhy,
  };
}

/* A red or amber light, an unserved stop sign or the destination inside
 * comfortable braking distance. */
function requiredStopNear(world) {
  const ego = world.ego;
  const front = ego.s + EGO.L / 2;
  const reach = (ego.v * ego.v) / (2 * 2.5) + 6;
  const near = (s) => s - front > -0.5 && s - front < reach;
  return LIGHTS.some((l) => near(l.s) && lightState(l, world.t) !== "green") ||
    SIGNS.some((g) => !ego.signs.has(g.id) && near(g.s)) ||
    near(MARKS.destination);
}

/* Every maneuver's outcome, each with a rank and a phrase key. */
export function plan(world) {
  const near = world.cars.filter((c) => interacts(c) && Math.abs(c.s - world.ego.s) < 250);
  const outcomes = OPTIONS.map((o) => rollout(world, o, near));
  const legal = outcomes.filter((o) => o.kind === null);
  let top = Math.max(0, ...legal.map((o) => o.progress));
  /* Close to a required stop, stopping is the maneuver: anything that only
   * creeps further is no better. */
  const stop = outcomes.find((o) => o.option === "stop");
  const forced = stop.kind === null && requiredStopNear(world);
  if (forced) top = stop.progress;
  for (const o of outcomes) {
    if (o.kind === null) {
      if (forced && o !== stop) o.kind = Math.abs(o.progress - top) < 0.5 ? "best" : "good";
      else if (forced) o.kind = "best";
      else if (top > 5 && o.moved < 0.5) o.kind = "idle";
      else if (o.progress >= top - Math.max(1, 0.03 * top)) o.kind = "best";
      else if (o.progress >= 0.6 * top) o.kind = "good";
      else o.kind = "slow";
    }
    o.rank = RANKS[o.kind];
    o.phrase = o.kind === "best"
      ? (o.stopWhy ?? (top < 1 ? "wait" : "best"))
      : o.kind === "illegal" ? o.why : o.kind;
  }
  return Object.fromEntries(outcomes.map((o) => [o.option, o]));
}

/* The state text. `random` picks phrasing and sentence order. */
export function describe(outcomes, random) {
  const pick = (list) => list[Math.floor(random() * list.length)];
  const sentences = OPTIONS.map((o) => `${o} ${pick(PHRASES[outcomes[o].phrase])}.`);
  for (let i = sentences.length - 1; i > 0; --i) {
    const j = Math.floor(random() * (i + 1));
    [sentences[i], sentences[j]] = [sentences[j], sentences[i]];
  }
  return sentences.join(" ");
}

/* Options with the best rank. */
export function best(outcomes) {
  const top = Math.min(...OPTIONS.map((o) => outcomes[o].rank));
  return OPTIONS.filter((o) => outcomes[o].rank === top);
}

/* The oracle's own pick among ties: most progress, then a fixed order. */
const PREFERENCE = ["keep", "faster", "slower", "stop", "right", "left"];
export function oracle(outcomes) {
  return best(outcomes).sort((a, b) =>
    outcomes[b].progress - outcomes[a].progress || PREFERENCE.indexOf(a) - PREFERENCE.indexOf(b))[0];
}

/* ------------------------------------------------------------- world -- */

export class World {
  constructor({ seed = 1, density = 1, start = 8, speed = 0, lane = null } = {}) {
    this.random = rng(seed);
    this.t = 0;
    this.lightOffsets = LIGHTS.map(() => this.random() * 22);
    const lanes = lanesAt(start).filter((l) => laneKind(start, l) === "drive");
    const l = lane ?? lanes[Math.floor(this.random() * lanes.length)];
    this.ego = {
      s: start, d: l * LANE_W, v: speed, a: 0, dv: 0, lane: l,
      signs: new Set(SIGNS.filter((g) => g.s < start + EGO.L).map((g) => g.id)),
      signWait: 0,
    };
    this.option = "keep";
    this.cars = spawnTraffic(this.random, density)
      .filter((c) => c.group !== "town" || c.s > start + 15 || c.s < start - 40)
      .filter((c) => !overlap(this.ego, EGO.L + 16, EGO.W, c));
    this.nextId = this.cars.length + 1;
    this.crashes = 0;
    this.violations = 0;
    this.events = [];
    this.arrived = false;
    this.offLane = false;
    this.applyLights();
  }

  applyLights() {
    LIGHTS.forEach((light, i) => (light.offset = this.lightOffsets[i]));
  }

  get zone() { return zoneAt(this.ego.s); }

  plan() { this.applyLights(); return plan(this); }

  /* Apply a decision: lateral maneuvers move the target lane once. */
  decide(option) {
    const lateral = option === "left" ? 1 : option === "right" ? -1 : 0;
    if (lateral && laneKind(this.ego.s + 3, this.ego.lane + lateral) !== null) {
      this.ego.lane += lateral;
    }
    this.option = option;
  }

  step(dt, manual = null) {
    this.applyLights();
    const ego = this.ego;
    const before = ego.s + EGO.L / 2;
    this.t += dt;
    stepTraffic(this.cars, ego, this.t, dt);
    if (manual) {
      ego.a = ego.v > 0 || manual.accel > 0 ? manual.accel : 0;
      ego.v = Math.max(0, Math.min(40, ego.v + manual.accel * dt));
      ego.s += ego.v * dt;
      const err = ego.lane * LANE_W - ego.d;
      ego.dv = Math.max(-1.6, Math.min(1.6, err * 1.6));
      ego.d += ego.dv * dt;
      for (const sign of SIGNS) {
        const g = sign.s - (ego.s + EGO.L / 2);
        if (!ego.signs.has(sign.id) && g < 4 && g > -1 && ego.v < 0.3) {
          ego.signWait += dt;
          if (ego.signWait >= 1) { ego.signs.add(sign.id); ego.signWait = 0; }
        }
      }
    } else {
      stepEgo(ego, this.option, this.cars, this.t, dt);
    }
    const front = ego.s + EGO.L / 2;

    for (const sign of SIGNS) {
      if (before < sign.s && front >= sign.s && !ego.signs.has(sign.id)) {
        ego.signs.add(sign.id);
        this.violation("rolled the stop sign");
      }
    }
    for (const light of LIGHTS) {
      if (before < light.s && front >= light.s && lightState(light, this.t) === "red") {
        this.violation("ran the red light");
      }
    }
    /* Off the end of a lane: count it and steer to the nearest real lane. */
    const off = lanesOverlapped(ego.d).some((l) => laneKind(front, l) !== "drive") &&
      laneKind(front, ego.lane) !== "drive";
    if (off && !this.offLane) {
      this.violation(laneKind(front, ego.lane) === "oncoming" ? "crossed into oncoming traffic"
        : ego.lane >= 0 && front > MARKS.exitEnd - 5 ? "missed the exit" : "ran out of lane");
      const lanes = lanesAt(front).filter((l) => laneKind(front, l) === "drive");
      if (lanes.length) ego.lane = lanes.reduce((a, b) => (Math.abs(b - ego.lane) < Math.abs(a - ego.lane) ? b : a));
    }
    this.offLane = off;

    for (const c of this.cars) {
      if (interacts(c) && overlap(ego, EGO.L, EGO.W, c)) {
        this.crashes += 1;
        this.events.push({ t: this.t, kind: "crash", s: ego.s, d: ego.d });
        ego.v = 0;
        this.respawn(c, true);
      }
    }
    this.recycle();
    if (!this.arrived && Math.abs(MARKS.destination - front) < 5 && ego.v < 0.3) {
      this.arrived = true;
      this.events.push({ t: this.t, kind: "arrived" });
    }
  }

  violation(what) {
    this.violations += 1;
    this.events.push({ t: this.t, kind: "violation", what });
  }

  /* Keep the interstate populated: cars that leave one end come back at the
   * other, clear of anything already there. */
  respawn(c, removeRoute = false) {
    if (c.group === "hw") {
      const ring = MARKS.hwEnd - MARKS.hwStart;
      let s = c.s > MARKS.hwEnd ? c.s - ring : c.s < MARKS.hwStart ? c.s + ring : null;
      if (s === null) {
        /* Collided in view: send it far away, the ego car's opposite side. */
        s = this.ego.s > (MARKS.hwStart + MARKS.hwEnd) / 2 ? MARKS.hwStart + 20 : MARKS.hwEnd - 20;
      }
      for (let tries = 0; tries < 40; ++tries) {
        const clash = this.cars.some((o) => o !== c && o.group === "hw" && o.lane === c.lane &&
          Math.abs(o.s - s) < (o.L + c.L) / 2 + 20);
        if (!clash) break;
        s -= 25;
      }
      c.s = s;
      c.v = c.v0;
      return;
    }
    if (removeRoute) c.gone = true;
  }

  recycle() {
    for (const c of this.cars) {
      if (c.group === "hw" && (c.s > MARKS.hwEnd || c.s < MARKS.hwStart)) this.respawn(c);
      else if (c.group === "oncoming" && c.s < c.range[0]) c.s = c.range[1];
      else if (c.group === "town" && c.s > MARKS.mergeStart - 30) c.gone = true;
      else if (c.group === "cedar" && c.s > ROUTE.length - 10) c.gone = true;
    }
    if (this.cars.some((c) => c.gone)) this.cars = this.cars.filter((c) => !c.gone);
  }

  /* A compact, JSON-friendly picture of the situation, for the inspector. */
  snapshot() {
    const ego = this.ego;
    const lead = leadOf(ego, this.cars, ego.lane * LANE_W);
    const light = LIGHTS.find((l) => l.s > ego.s - 5 && l.s - ego.s < 150);
    const sign = SIGNS.find((g) => !ego.signs.has(g.id) && g.s > ego.s && g.s - ego.s < 150);
    return {
      zone: ZONE_NAMES[this.zone],
      speed_kmh: Math.round(ego.v * 3.6),
      limit_kmh: Math.round(limitAt(ego.s) * 3.6),
      lane: ego.lane,
      route_lanes: desiredLanes(ego.s),
      lead: lead ? { gap_m: +lead.gap.toFixed(1), speed_kmh: Math.round(lead.car.v * 3.6) } : null,
      light: light ? { in_m: Math.round(light.s - ego.s), state: lightState(light, this.t) } : null,
      stop_sign: sign ? { in_m: Math.round(sign.s - ego.s) } : null,
      destination_m: Math.round(MARKS.destination - ego.s),
    };
  }
}
