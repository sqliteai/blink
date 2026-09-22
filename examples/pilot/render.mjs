/* render.mjs -- top-down drawing of the world in world.mjs, for index.html.
 *
 * Everything static (roads, markings, scenery, the minimap) is built once
 * from the route. Each frame draws it in a camera that follows the ego car
 * with its heading up, then the candidate paths, the traffic and the HUD.
 *
 * SPDX-License-Identifier: Apache-2.0 */

import {
  EGO, LANE_W, LIGHTS, MARKS, ROUTE, SIGNS, ZONE_NAMES, highwayPose, laneKind,
  lanesAt, lightState, limitAt, pose, rng,
} from "./world.mjs";

const COLORS = {
  grass: "#5d8a4a", grassDark: "#527d41", asphalt: "#3a3d42", asphaltEdge: "#2c2f33",
  white: "rgba(245,245,240,0.92)", yellow: "#e8c547", ego: "#f4f6f8",
};
/* Candidate colours by outcome rank: best, fine, poor, illegal/crash. */
const RANK_COLOR = (rank) =>
  rank === 0 ? "#34d1ff" : rank <= 3 ? "#5b8cff" : rank <= 6 ? "#ffb020" : rank <= 9 ? "#ff5a36" : "#b06bff";

/* ------------------------------------------------------ static geometry -- */

function bandPolygon(s0, s1, d0, d1, at = pose) {
  const pts = [];
  for (let s = s0; s <= s1; s += 2) { const p = at(s, d1); pts.push([p.x, p.y]); }
  const e = at(s1, d1); pts.push([e.x, e.y]);
  for (let s = s1; s >= s0; s -= 2) { const p = at(s, d0); pts.push([p.x, p.y]); }
  const b = at(s0, d0); pts.push([b.x, b.y]);
  return pts;
}
function linePoints(s0, s1, d, at = pose) {
  const pts = [];
  for (let s = s0; s <= s1; s += 2) { const p = at(s, d); pts.push([p.x, p.y]); }
  const e = at(s1, d); pts.push([e.x, e.y]);
  return pts;
}

function buildStatic() {
  const bands = [], lines = [];
  for (const r of ROUTE.zoneRanges) {
    const lanes = lanesAt(r.start + 1);
    const lo = lanes[0], hi = lanes[lanes.length - 1];
    /* The town roads carry on past both ends of the trip. */
    const a = r.start === 0 ? -120 : r.start, z = r.end >= ROUTE.length ? ROUTE.length + 120 : r.end;
    bands.push(bandPolygon(a - 1, z + 1, (lo - 0.5) * LANE_W, (hi + 0.5) * LANE_W));
    lines.push({ pts: linePoints(a, z, (lo - 0.5) * LANE_W), style: "edge" });
    lines.push({ pts: linePoints(a, z, (hi + 0.5) * LANE_W), style: "edge" });
    for (let l = lo; l < hi; ++l) {
      const ka = laneKind(r.start + 1, l), kb = laneKind(r.start + 1, l + 1);
      const style = ka === "oncoming" || kb === "oncoming" ? "center" : "dash";
      lines.push({ pts: linePoints(a, z, (l + 0.5) * LANE_W), style });
    }
  }
  /* The interstate beyond the stretch the route uses. */
  for (const [s0, s1] of [[MARKS.hwStart - 80, MARKS.mergeStart], [MARKS.exitEnd, MARKS.hwEnd + 80]]) {
    bands.push(bandPolygon(s0, s1, -0.5 * LANE_W, 2.5 * LANE_W, highwayPose));
    lines.push({ pts: linePoints(s0, s1, -0.5 * LANE_W, highwayPose), style: "edge" });
    lines.push({ pts: linePoints(s0, s1, 2.5 * LANE_W, highwayPose), style: "edge" });
    for (const d of [0.5, 1.5]) lines.push({ pts: linePoints(s0, s1, d * LANE_W, highwayPose), style: "dash" });
  }
  /* The merge and the exit: a solid line where the ramp runs alongside. */
  lines.push({ pts: linePoints(MARKS.mergeStart - 140, MARKS.mergeStart, -0.5 * LANE_W, highwayPose), style: "edge" });

  /* Cross streets at every sign and light. */
  const cross = [];
  for (const s of [...SIGNS.map((g) => g.s), ...LIGHTS.map((l) => l.s)]) {
    const c = pose(s + 6, -LANE_W / 2);
    cross.push({ x: c.x, y: c.y, h: c.h + Math.PI / 2, len: 220, w: 9 });
  }

  /* Scenery, kept clear of every road. */
  const random = rng(4242);
  const roadPts = [];
  for (let s = -120; s <= ROUTE.length + 120; s += 4) {
    const lanes = lanesAt(s);
    const mid = ((lanes[0] + lanes[lanes.length - 1]) / 2) * LANE_W;
    const p = pose(s, mid);
    roadPts.push([p.x, p.y, ((lanes[lanes.length - 1] - lanes[0] + 1) * LANE_W) / 2]);
  }
  for (let s = MARKS.hwStart - 80; s <= MARKS.hwEnd + 80; s += 4) {
    const p = highwayPose(s, LANE_W);
    roadPts.push([p.x, p.y, 1.5 * LANE_W]);
  }
  for (const c of cross) {
    for (let k = -c.len / 2; k <= c.len / 2; k += 4) {
      roadPts.push([c.x + Math.cos(c.h) * k, c.y + Math.sin(c.h) * k, c.w / 2]);
    }
  }
  const clear = (x, y, r) => roadPts.every(([px, py, hw]) => Math.hypot(px - x, py - y) > hw + r + 2.5);
  const trees = [], buildings = [];
  for (let s = 0; s <= ROUTE.length; s += 7) {
    const zone = ROUTE.zones[Math.floor(s)];
    const town = zone === "town" || zone === "cedar";
    for (const side of [-1, 1]) {
      if (town && random() < 0.55) {
        const w = 9 + random() * 10, depth = 8 + random() * 8;
        const off = side * (11 + depth / 2 + random() * 6);
        const p = pose(s, off - LANE_W / 2);
        if (clear(p.x, p.y, Math.hypot(w, depth) / 2)) {
          buildings.push({ x: p.x, y: p.y, h: p.h, w, depth,
            color: ["#c7b8a3", "#b9a58c", "#d4cbbd", "#a89f94", "#c9a88e"][Math.floor(random() * 5)],
            roof: random() < 0.5 });
          continue;
        }
      }
      const n = town ? 1 : random() < 0.6 ? 2 : 0;
      for (let k = 0; k < n; ++k) {
        const off = side * (10 + random() * (town ? 20 : 45));
        const p = pose(s + random() * 7, off);
        const r = 1.8 + random() * 2.2;
        if (clear(p.x, p.y, r)) trees.push({ x: p.x, y: p.y, r, tone: random() });
      }
    }
  }
  /* Trees along the interstate beyond the route. */
  for (let s = MARKS.hwStart - 60; s <= MARKS.hwEnd + 60; s += 9) {
    for (const side of [-1, 1]) {
      if (random() < 0.5) continue;
      const p = highwayPose(s, LANE_W + side * (12 + random() * 40));
      const r = 1.8 + random() * 2.2;
      if (clear(p.x, p.y, r)) trees.push({ x: p.x, y: p.y, r, tone: random() });
    }
  }
  return { bands, lines, cross, trees, buildings };
}

function buildMinimap(size) {
  const pts = [];
  for (let s = 0; s <= ROUTE.length; s += 5) { const p = pose(s, -LANE_W / 2); pts.push([p.x, p.y]); }
  const hw = [];
  for (let s = MARKS.hwStart - 80; s <= MARKS.hwEnd + 80; s += 10) { const p = highwayPose(s, LANE_W); hw.push([p.x, p.y]); }
  const all = [...pts, ...hw];
  const minX = Math.min(...all.map((p) => p[0])), maxX = Math.max(...all.map((p) => p[0]));
  const minY = Math.min(...all.map((p) => p[1])), maxY = Math.max(...all.map((p) => p[1]));
  const pad = 12;
  const k = (size - 2 * pad) / Math.max(maxX - minX, maxY - minY);
  const ox = pad + ((size - 2 * pad) - (maxX - minX) * k) / 2, oy = pad + ((size - 2 * pad) - (maxY - minY) * k) / 2;
  const map = (x, y) => [ox + (x - minX) * k, oy + (y - minY) * k];
  return { pts: pts.map((p) => map(...p)), hw: hw.map((p) => map(...p)), map };
}

/* --------------------------------------------------------------- frame -- */

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.geo = buildStatic();
    this.mini = buildMinimap(150);
    this.camH = null;
    this.flash = 0;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.dpr = dpr;
  }

  path(pts, close = false) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; ++i) ctx.lineTo(pts[i][0], pts[i][1]);
    if (close) ctx.closePath();
  }

  draw(world, view) {
    const { ctx, canvas } = this;
    const W = canvas.width / this.dpr, H = canvas.height / this.dpr;
    const ego = world.ego;
    const p = pose(ego.s, ego.d);
    const heading = p.h + Math.atan2(ego.dv ?? 0, Math.max(ego.v, 2));
    if (this.camH === null) this.camH = p.h;
    let dh = p.h - this.camH;
    dh = Math.atan2(Math.sin(dh), Math.cos(dh));
    this.camH += dh * Math.min(1, view.dt * 3);
    const ppm = Math.min(W, H) / (36 + ego.v * 1.0);

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = COLORS.grass;
    ctx.fillRect(0, 0, W, H);
    ctx.save();
    ctx.translate(W / 2, H * 0.66);
    ctx.scale(ppm, ppm);
    ctx.rotate(-Math.PI / 2 - this.camH);
    ctx.translate(-p.x, -p.y);
    const radius = Math.hypot(W, H) / ppm;
    const near = (x, y, r = 0) => Math.abs(x - p.x) < radius + r && Math.abs(y - p.y) < radius + r;

    /* Roads. */
    ctx.fillStyle = COLORS.asphalt;
    for (const c of this.geo.cross) {
      if (!near(c.x, c.y, c.len)) continue;
      ctx.save(); ctx.translate(c.x, c.y); ctx.rotate(c.h);
      ctx.fillRect(-c.len / 2, -c.w / 2, c.len, c.w);
      ctx.restore();
    }
    for (const b of this.geo.bands) { this.path(b, true); ctx.fill(); }
    ctx.lineCap = "butt";
    for (const l of this.geo.lines) {
      if (l.style === "dash") { ctx.setLineDash([3, 6]); ctx.strokeStyle = COLORS.white; ctx.lineWidth = 0.15; }
      else if (l.style === "center") { ctx.setLineDash([]); ctx.strokeStyle = COLORS.yellow; ctx.lineWidth = 0.35; }
      else { ctx.setLineDash([]); ctx.strokeStyle = COLORS.white; ctx.lineWidth = 0.18; }
      this.path(l.pts); ctx.stroke();
    }
    ctx.setLineDash([]);

    /* Stop lines, signs and signals. */
    const stopLine = (s, color) => {
      const a = pose(s, -1.5 * LANE_W), b = pose(s, -0.5 * LANE_W);
      ctx.strokeStyle = color; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    };
    for (const sign of SIGNS) {
      const q = pose(sign.s - 1, -1.5 * LANE_W - 1.6);
      if (!near(q.x, q.y)) continue;
      stopLine(sign.s, COLORS.white);
      ctx.save(); ctx.translate(q.x, q.y); ctx.rotate(this.camH + Math.PI / 2);
      ctx.fillStyle = ego.signs.has(sign.id) ? "#8c3b33" : "#d7322a";
      ctx.beginPath();
      for (let k = 0; k < 8; ++k) {
        const a = Math.PI / 8 + (k * Math.PI) / 4;
        ctx.lineTo(Math.cos(a) * 1.1, Math.sin(a) * 1.1);
      }
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 0.12; ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.font = "bold 0.55px system-ui, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("STOP", 0, 0.03);
      ctx.restore();
    }
    for (const light of LIGHTS) {
      const q = pose(light.s - 0.5, -1.5 * LANE_W - 1.6);
      if (!near(q.x, q.y)) continue;
      const state = lightState(light, world.t);
      stopLine(light.s, state === "green" ? COLORS.white : state === "amber" ? "#ffcf4a" : "#ff5a4a");
      ctx.save(); ctx.translate(q.x, q.y); ctx.rotate(this.camH + Math.PI / 2);
      ctx.fillStyle = "#16181b";
      ctx.fillRect(-0.7, -1.9, 1.4, 3.8);
      [["red", -1.15, "#ff4b3e"], ["amber", 0, "#ffc02e"], ["green", 1.15, "#3ee07a"]].forEach(([name, y, on]) => {
        ctx.fillStyle = state === name ? on : "#3a3d42";
        ctx.beginPath(); ctx.arc(0, y, 0.48, 0, Math.PI * 2); ctx.fill();
      });
      ctx.restore();
    }

    /* Destination. */
    const dest = pose(MARKS.destination, -LANE_W);
    if (near(dest.x, dest.y)) {
      ctx.save(); ctx.translate(dest.x, dest.y); ctx.rotate(dest.h);
      for (let i = 0; i < 6; ++i) for (let j = 0; j < 4; ++j) {
        ctx.fillStyle = (i + j) % 2 ? "#111" : "#f5f5f5";
        ctx.fillRect(-0.3 + i * 0.1 - 0.3, -1.6 + j * 0.8, 0.1, 0.8);
      }
      ctx.restore();
      const pin = pose(MARKS.destination + 2, -2 * LANE_W - 1);
      ctx.fillStyle = "#ff5a36";
      ctx.beginPath(); ctx.arc(pin.x, pin.y, 1.4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.beginPath(); ctx.arc(pin.x, pin.y, 0.55, 0, Math.PI * 2); ctx.fill();
    }

    /* Scenery. */
    for (const b of this.geo.buildings) {
      if (!near(b.x, b.y, 20)) continue;
      ctx.save(); ctx.translate(b.x, b.y); ctx.rotate(b.h);
      ctx.fillStyle = "rgba(0,0,0,0.18)";
      ctx.fillRect(-b.w / 2 + 0.8, -b.depth / 2 + 0.8, b.w, b.depth);
      ctx.fillStyle = b.color;
      ctx.fillRect(-b.w / 2, -b.depth / 2, b.w, b.depth);
      ctx.strokeStyle = "rgba(0,0,0,0.15)"; ctx.lineWidth = 0.3;
      if (b.roof) { ctx.beginPath(); ctx.moveTo(-b.w / 2, 0); ctx.lineTo(b.w / 2, 0); ctx.stroke(); }
      ctx.strokeRect(-b.w / 2, -b.depth / 2, b.w, b.depth);
      ctx.restore();
    }

    /* Candidate paths: what each maneuver would do over the next 3 s. */
    if (view.candidates && view.outcomes) {
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      const order = Object.values(view.outcomes).sort((a, b) => b.rank - a.rank);
      for (const o of order) {
        if (o.path.length < 2) continue;
        const pts = o.path.map(([s, d]) => { const q = pose(s, d); return [q.x, q.y]; });
        const picked = o.option === view.choice;
        const prob = view.probabilities?.[o.option] ?? 0;
        ctx.globalAlpha = picked ? 0.95 : 0.35 + 0.5 * prob;
        ctx.strokeStyle = picked ? "#9ff7ff" : RANK_COLOR(o.rank);
        ctx.lineWidth = picked ? 0.9 : 0.4;
        this.path(pts); ctx.stroke();
        const end = pts[pts.length - 1];
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath(); ctx.arc(end[0], end[1], picked ? 0.7 : 0.45, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    /* Traffic. */
    for (const c of world.cars) {
      let q;
      if (c.group === "hw") q = highwayPose(c.s, c.d);
      else if (c.group === "oncoming") { q = pose(c.s, c.d); q.h += Math.PI; }
      else q = pose(c.s, c.d);
      if (!near(q.x, q.y, 15)) continue;
      drawCar(ctx, q.x, q.y, q.h, c.L, c.W, c.color, c.a < -1 && c.v > 0.1 || c.v < 0.1 && c.group !== "oncoming", c.truck);
    }
    drawCar(ctx, p.x, p.y, heading, EGO.L, EGO.W, COLORS.ego, (ego.a ?? 0) < -1 || ego.v < 0.1, false, true);

    /* Trees above the cars. */
    for (const t of this.geo.trees) {
      if (!near(t.x, t.y, 5)) continue;
      ctx.fillStyle = "rgba(0,0,0,0.2)";
      ctx.beginPath(); ctx.arc(t.x + 0.7, t.y + 0.7, t.r, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = t.tone < 0.5 ? "#2f6b34" : "#3b7a3a";
      ctx.beginPath(); ctx.arc(t.x, t.y, t.r, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.beginPath(); ctx.arc(t.x - t.r * 0.3, t.y - t.r * 0.3, t.r * 0.55, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();

    if (this.flash > 0) {
      ctx.fillStyle = `rgba(255,60,40,${0.35 * this.flash})`;
      ctx.fillRect(0, 0, W, H);
      this.flash = Math.max(0, this.flash - view.dt * 1.5);
    }
    this.hud(world, view, W, H);
  }

  hud(world, view, W, H) {
    const ctx = this.ctx, ego = world.ego;
    /* Speed and limit. */
    ctx.fillStyle = "rgba(15,17,20,0.72)";
    roundRect(ctx, 14, H - 92, 178, 78, 12); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.font = "600 40px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(String(Math.round(ego.v * 3.6)), 28, H - 42);
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillText("km/h", 30, H - 24);
    const limit = Math.round(limitAt(ego.s) * 3.6);
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(155, H - 53, 23, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#d7322a"; ctx.lineWidth = 5;
    ctx.beginPath(); ctx.arc(155, H - 53, 20.5, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "#111"; ctx.textAlign = "center"; ctx.font = "700 16px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(String(limit), 155, H - 47);

    /* Mode and place. */
    const auto = view.autopilot;
    ctx.textAlign = "left";
    ctx.font = "600 13px ui-sans-serif, system-ui, sans-serif";
    const label = auto ? `AUTOPILOT · ${view.policy === "blink" ? "Blink" : "Oracle"}` : "MANUAL · WASD";
    const w = ctx.measureText(label).width + 26;
    ctx.fillStyle = auto ? "rgba(40,120,255,0.92)" : "rgba(15,17,20,0.72)";
    roundRect(ctx, 14, 14, w, 30, 15); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.fillText(label, 27, 34);
    ctx.fillStyle = "rgba(15,17,20,0.72)";
    const zone = ZONE_NAMES[world.zone];
    const zw = ctx.measureText(zone).width + 26;
    roundRect(ctx, 14, 50, zw, 28, 14); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "13px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(zone, 27, 69);
    if (view.choice) {
      const text = `▸ ${view.choice}`;
      ctx.font = "600 13px ui-sans-serif, system-ui, sans-serif";
      const cw = ctx.measureText(text).width + 26;
      ctx.fillStyle = "rgba(15,17,20,0.72)";
      roundRect(ctx, 14, 84, cw, 28, 14); ctx.fill();
      ctx.fillStyle = "#9ff7ff"; ctx.fillText(text, 27, 103);
    }

    /* Minimap. */
    const m = this.mini, size = 150, x0 = W - size - 14, y0 = 14;
    ctx.fillStyle = "rgba(15,17,20,0.72)";
    roundRect(ctx, x0, y0, size, size, 12); ctx.fill();
    ctx.save(); ctx.translate(x0, y0);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(255,255,255,0.28)"; ctx.lineWidth = 4;
    this.path(m.hw); ctx.stroke();
    ctx.strokeStyle = "rgba(120,190,255,0.9)"; ctx.lineWidth = 2.5;
    this.path(m.pts); ctx.stroke();
    const d = pose(MARKS.destination, -LANE_W);
    const [dx, dy] = m.map(d.x, d.y);
    ctx.fillStyle = "#ff5a36"; ctx.beginPath(); ctx.arc(dx, dy, 4, 0, Math.PI * 2); ctx.fill();
    const e = pose(ego.s, ego.d);
    const [ex, ey] = m.map(e.x, e.y);
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(ex, ey, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#2878ff"; ctx.lineWidth = 2; ctx.stroke();
    ctx.restore();

    if (view.banner) {
      ctx.fillStyle = "rgba(10,12,14,0.78)";
      roundRect(ctx, W / 2 - 190, H / 2 - 44, 380, 88, 14); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.textAlign = "center";
      ctx.font = "600 20px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(view.banner[0], W / 2, H / 2 - 6);
      ctx.font = "14px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.fillText(view.banner[1], W / 2, H / 2 + 20);
    }
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawCar(ctx, x, y, h, L, W, color, braking, truck, ego = false) {
  ctx.save();
  ctx.translate(x, y); ctx.rotate(h);
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  roundRect(ctx, -L / 2 + 0.3, -W / 2 + 0.35, L, W, 0.5); ctx.fill();
  if (truck) {
    ctx.fillStyle = color;
    roundRect(ctx, -L / 2, -W / 2, L - 2.6, W, 0.25); ctx.fill();
    ctx.fillStyle = "#3b5fa8";
    roundRect(ctx, L / 2 - 2.4, -W / 2 + 0.05, 2.4, W - 0.1, 0.4); ctx.fill();
    ctx.fillStyle = "#1c2330";
    ctx.fillRect(L / 2 - 0.9, -W / 2 + 0.25, 0.5, W - 0.5);
  } else {
    ctx.fillStyle = color;
    roundRect(ctx, -L / 2, -W / 2, L, W, 0.7); ctx.fill();
    ctx.fillStyle = "rgba(20,26,34,0.85)";
    roundRect(ctx, L * 0.06, -W / 2 + 0.22, L * 0.2, W - 0.44, 0.25); ctx.fill();
    roundRect(ctx, -L * 0.36, -W / 2 + 0.28, L * 0.13, W - 0.56, 0.2); ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.14)";
    roundRect(ctx, -L * 0.22, -W / 2 + 0.25, L * 0.27, W - 0.5, 0.3); ctx.fill();
  }
  ctx.fillStyle = "#fff6c8";
  ctx.fillRect(L / 2 - 0.18, -W / 2 + 0.15, 0.18, 0.4);
  ctx.fillRect(L / 2 - 0.18, W / 2 - 0.55, 0.18, 0.4);
  ctx.fillStyle = braking ? "#ff2a1a" : "#7a1a14";
  ctx.fillRect(-L / 2, -W / 2 + 0.12, 0.2, 0.45);
  ctx.fillRect(-L / 2, W / 2 - 0.57, 0.2, 0.45);
  if (ego) {
    ctx.strokeStyle = "#2878ff"; ctx.lineWidth = 0.18;
    roundRect(ctx, -L / 2, -W / 2, L, W, 0.7); ctx.stroke();
  }
  ctx.restore();
}
