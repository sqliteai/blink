/* render.mjs -- draws a Doom game from doom.mjs at ViZDoom's 320x240.
 *
 * A raycaster for the rectangular rooms, textured walls and floor, and every
 * sprite drawn procedurally: no WAD, no id Software or Freedoom art. The
 * monster is projected with the same function that writes the label box the
 * model reads, so the box drawn over it is exactly what Blink sees.
 *
 * Browser only. SPDX-License-Identifier: Apache-2.0 */

import { FOCAL_X, FOCAL_Y, SCREEN_H, SCREEN_W, SPRITE_H, SPRITE_W } from "./doom.mjs";

const EYE = 41, CEILING = 128;
const RAD = Math.PI / 180;
const clamp = (v, lo = 0, hi = 1) => (v < lo ? lo : v > hi ? hi : v);

export class Renderer {
  constructor(canvas) {
    this.ctx = canvas.getContext("2d");
    this.image = this.ctx.createImageData(SCREEN_W, SCREEN_H);
    this.pixels = this.image.data;
    this.depth = new Float32Array(SCREEN_W);
  }

  draw(game, { showBox = true } = {}) {
    this.game = game;
    this.world(game);
    this.sprites(game);
    this.weapon(game);
    this.ctx.putImageData(this.image, 0, 0);
    if (showBox) this.box(game);
  }

  put(x, y, r, g, b) {
    const i = (y * SCREEN_W + x) * 4;
    this.pixels[i] = r; this.pixels[i + 1] = g; this.pixels[i + 2] = b; this.pixels[i + 3] = 255;
  }

  blend(x, y, r, g, b, a) {
    const i = (y * SCREEN_W + x) * 4, p = this.pixels;
    p[i] += (r - p[i]) * a; p[i + 1] += (g - p[i + 1]) * a; p[i + 2] += (b - p[i + 2]) * a;
  }

  /* Walls, floor and ceiling, one column at a time. */
  world(game) {
    const { room } = game.spec, p = game.player;
    const nukage = game.scenario === "predict_position";
    const cos = Math.cos(p.angle * RAD), sin = Math.sin(p.angle * RAD);
    for (let x = 0; x < SCREEN_W; ++x) {
      const offset = (SCREEN_W / 2 - x - 0.5) / FOCAL_X;       /* left of centre is positive */
      const dx = cos - sin * offset, dy = sin + cos * offset;   /* ray, unit depth */
      let t = Infinity, u = 0, side = 0;
      if (dx > 0) { const k = (room.x1 - p.x) / dx; if (k < t) { t = k; u = p.y + dy * k; side = 0; } }
      if (dx < 0) { const k = (room.x0 - p.x) / dx; if (k < t) { t = k; u = p.y + dy * k; side = 0; } }
      if (dy > 0) { const k = (room.y1 - p.y) / dy; if (k < t) { t = k; u = p.x + dx * k; side = 1; } }
      if (dy < 0) { const k = (room.y0 - p.y) / dy; if (k < t) { t = k; u = p.x + dx * k; side = 1; } }
      this.depth[x] = t;                                         /* depth along the view axis */
      const top = SCREEN_H / 2 - FOCAL_Y * (CEILING - EYE) / t;
      const bottom = SCREEN_H / 2 + FOCAL_Y * EYE / t;
      const light = clamp(1.25 - t / 900, 0.25, 1) * (side ? 0.82 : 1);
      for (let y = 0; y < SCREEN_H; ++y) {
        if (y < top) {
          /* Ceiling: dark grey panels. */
          const d = FOCAL_Y * (CEILING - EYE) / (SCREEN_H / 2 - y + 0.5);
          const wx = p.x + dx * d, wy = p.y + dy * d;
          const seam = ((wx & 63) < 2 || (wy & 63) < 2) ? 0.7 : 1;
          const l = clamp(1.2 - d / 700, 0.2, 1) * seam;
          this.put(x, y, 72 * l, 70 * l, 66 * l);
        } else if (y < bottom) {
          /* Wall: brown brick, ViZDoom's default look. */
          const v = CEILING - (y - top) / (bottom - top) * CEILING;
          const row = Math.floor(v / 16);
          const shift = row & 1 ? 16 : 0;
          const mortar = (v % 16) < 1.6 || ((u + shift) % 32 + 32) % 32 < 1.6;
          const grain = 0.9 + 0.1 * Math.sin(u * 0.9 + row * 7.1);
          const l = light * (mortar ? 0.55 : grain);
          this.put(x, y, 150 * l, 104 * l, 70 * l);
        } else {
          /* Floor: grey flagstones, or green slime in Predict Position. */
          const d = FOCAL_Y * EYE / (y - SCREEN_H / 2 + 0.5);
          const wx = p.x + dx * d, wy = p.y + dy * d;
          const l = clamp(1.2 - d / 700, 0.18, 1);
          if (nukage) {
            const ripple = 0.8 + 0.2 * Math.sin(wx * 0.08 + game.ticks * 0.15) * Math.cos(wy * 0.07);
            this.put(x, y, 40 * l * ripple, 110 * l * ripple, 40 * l * ripple);
          } else {
            const seam = ((wx & 31) < 1.5 || (wy & 31) < 1.5) ? 0.6 : 1;
            const tone = 0.85 + 0.15 * (((wx >> 5) + (wy >> 5)) & 1);
            this.put(x, y, 96 * l * seam * tone, 92 * l * seam * tone, 86 * l * seam * tone);
          }
        }
      }
    }
  }

  /* A world point to screen: centre x, y and the scale at that depth. */
  project(wx, wy, drop) {
    const p = this.game.player, a = p.angle * RAD;
    const dx = wx - p.x, dy = wy - p.y;
    const depth = dx * Math.cos(a) + dy * Math.sin(a);
    if (depth < 8) return null;
    const lateral = dx * Math.sin(a) - dy * Math.cos(a);
    return { x: SCREEN_W / 2 + FOCAL_X * lateral / depth, y: SCREEN_H / 2 + FOCAL_Y * drop / depth,
      sx: FOCAL_X / depth, sy: FOCAL_Y / depth, depth };
  }

  /* Paint a sprite: `shade(u, v)` gets coordinates in [-1, 1] and returns
   * [r, g, b, alpha] or null for a transparent pixel. */
  paint(at, width, height, shade) {
    const w = width * at.sx, h = height * at.sy;
    const x0 = Math.max(0, Math.floor(at.x - w / 2)), x1 = Math.min(SCREEN_W, Math.ceil(at.x + w / 2));
    const y0 = Math.max(0, Math.floor(at.y - h / 2)), y1 = Math.min(SCREEN_H, Math.ceil(at.y + h / 2));
    const light = clamp(1.25 - at.depth / 900, 0.3, 1);
    for (let x = x0; x < x1; ++x) {
      if (at.depth > this.depth[x]) continue;
      const u = ((x + 0.5 - at.x) / w) * 2;
      for (let y = y0; y < y1; ++y) {
        const c = shade(u, ((y + 0.5 - at.y) / h) * 2);
        if (!c) continue;
        const a = c[3] ?? 1, l = c[4] ? 1 : light;              /* c[4]: emissive */
        if (a >= 1) this.put(x, y, c[0] * l, c[1] * l, c[2] * l);
        else this.blend(x, y, c[0] * l, c[1] * l, c[2] * l, a);
      }
    }
  }

  sprites(game) {
    const list = [];
    const m = game.monster;
    if (m.alive || m.dying < 24) list.push({ x: m.x, y: m.y, draw: (at) => this.cacodemon(at, m) });
    for (const s of game.shots) list.push({ x: s.x, y: s.y, draw: (at) => this.puff(at, s) });
    if (game.rocket) list.push({ x: game.rocket.x, y: game.rocket.y, draw: (at) => this.rocket(at) });
    if (game.explosion) list.push({ x: game.explosion.x, y: game.explosion.y, draw: (at) => this.explosion(at, game.explosion) });
    const p = game.player;
    list.sort((a, b) => Math.hypot(b.x - p.x, b.y - p.y) - Math.hypot(a.x - p.x, a.y - p.y));
    for (const s of list) {
      const at = this.project(s.x, s.y, game.spec.spriteDrop);
      if (at) s.draw(at);
    }
  }

  /* Red, one eye, horns, a mouth full of teeth. Drawn, not ripped. */
  cacodemon(at, m) {
    const dying = m.alive ? 0 : m.dying / 24;
    const sink = dying * 0.9;
    this.paint({ ...at, y: at.y + sink * SPRITE_H * at.sy * 0.5 }, SPRITE_W, SPRITE_H * (1 - dying * 0.5), (u, v) => {
      const r2 = u * u + v * v * 1.1;
      /* Horns */
      for (const hx of [-0.5, 0.5]) {
        if (v < -0.55 && v > -1 && Math.abs(u - hx) < (v + 1) * 0.28) return [225, 205, 170];
      }
      if (r2 > 0.8) return null;
      const nz = Math.sqrt(1 - r2 / 0.8);
      const lit = clamp(0.3 + 0.55 * nz + 0.25 * (-u * 0.5 - v * 0.7));
      const burn = 1 - dying * 0.7;
      /* Eye */
      const ex = u, ey = v + 0.2, er = ex * ex + ey * ey;
      if (er < 0.075 && !dying) {
        if (er < 0.012) return [10, 10, 10];
        if (er < 0.05) return [60, 210, 70, 1, 1];
        return [240, 240, 220];
      }
      /* Mouth and teeth */
      const mx = u / 0.5, my = (v - 0.42) / 0.2;
      if (mx * mx + my * my < 1) {
        const tooth = (my < -0.2 && (Math.floor((u + 1) * 11) & 1)) || (my > 0.45 && (Math.floor((u + 1) * 11) & 1) === 0);
        return tooth ? [235, 230, 210] : [70 * burn, 5, 10];
      }
      /* Blotches on the skin */
      const spot = Math.sin(u * 13) * Math.sin(v * 11) > 0.72 ? 0.8 : 1;
      return [205 * lit * spot * burn, 38 * lit * spot * burn, 30 * lit * spot * burn];
    });
  }

  puff(at, shot) {
    const s = 10 + shot.age * 2, fade = 1 - shot.age / 6;
    this.paint(at, s, s, (u, v) => {
      const r2 = u * u + v * v;
      if (r2 > 1) return null;
      return shot.blood ? [170, 0, 0, fade] : [200, 200, 190, fade * (1 - r2)];
    });
  }

  rocket(at) {
    this.paint(at, 18, 18, (u, v) => {
      const r2 = u * u + v * v;
      if (r2 > 1) return null;
      if (r2 < 0.25) return [255, 250, 200, 1, 1];
      return [255, 150, 40, 1 - r2, 1];
    });
  }

  explosion(at, e) {
    const size = 30 + e.age * 8, fade = 1 - e.age / 13;
    this.paint(at, size, size, (u, v) => {
      const r2 = u * u + v * v;
      if (r2 > 1) return null;
      const hot = 1 - r2;
      return [255, 120 + 130 * hot, 40 * hot, fade * (0.4 + 0.6 * hot), 1];
    });
  }

  /* The gun, bobbing while the player strafes, with a muzzle flash. */
  weapon(game) {
    const w = game.weapon, launcher = game.scenario === "predict_position";
    const bob = Math.round(Math.abs(Math.sin(game.ticks * 0.3)) * Math.min(6, Math.abs(game.player.vy)));
    const recoil = w.flash > 0 ? 4 : 0;
    const cx = SCREEN_W / 2 + (launcher ? 20 : 8), base = SCREEN_H + recoil + bob;
    const rect = (x0, y0, x1, y1, r, g, b) => {
      for (let x = Math.max(0, x0); x < Math.min(SCREEN_W, x1); ++x)
        for (let y = Math.max(0, y0); y < Math.min(SCREEN_H, y1); ++y) {
          const edge = x === x0 || x === x1 - 1 ? 0.7 : 1;
          this.put(x, y, r * edge, g * edge, b * edge);
        }
    };
    if (w.flash > 0) {
      const fy = base - (launcher ? 78 : 70), fr = 10 + w.flash * 2;
      for (let x = cx - fr; x < cx + fr; ++x)
        for (let y = fy - fr; y < fy + fr; ++y) {
          const d = Math.hypot(x - cx, y - fy) / fr;
          if (d < 1 && x >= 0 && y >= 0 && x < SCREEN_W && y < SCREEN_H) this.blend(x, y, 255, 230, 120, 1 - d);
        }
    }
    if (launcher) {
      rect(cx - 18, base - 72, cx + 18, base, 70, 78, 64);
      rect(cx - 14, base - 76, cx + 14, base - 68, 40, 44, 38);
      rect(cx - 30, base - 30, cx + 30, base, 90, 96, 80);
      rect(cx - 8, base - 22, cx + 14, base, 200, 150, 120);
    } else {
      rect(cx - 6, base - 66, cx + 6, base - 30, 110, 110, 116);
      rect(cx - 10, base - 34, cx + 10, base - 6, 80, 80, 86);
      rect(cx - 16, base - 22, cx + 16, base, 205, 155, 125);
    }
  }

  /* The label box, exactly as the state text reports it. */
  box(game) {
    const b = game.monster.alive ? game.project() : null;
    if (!b) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(90, 255, 140, 0.9)";
    ctx.lineWidth = 1;
    ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
    ctx.fillStyle = "rgba(90, 255, 140, 0.95)";
    ctx.font = "9px ui-monospace, Menlo, monospace";
    const label = `x ${Math.floor(b.x + b.w / 2)} w ${b.w}`;
    ctx.fillText(label, Math.min(SCREEN_W - 52, Math.max(1, b.x)), b.y > 12 ? b.y - 3 : b.y + b.h + 9);
    ctx.restore();
  }
}
