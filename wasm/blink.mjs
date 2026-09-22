/* blink.mjs -- JavaScript binding for the WebAssembly build of the runtime.
 *
 *   import { loadBlink } from "./blink.mjs";
 *   const blink   = await loadBlink(wasmBytes);          // or a URL / Response
 *   const model   = blink.openModel(containerBytes);     // Uint8Array
 *   const session = model.createSession({ maxOptions: 4 });
 *   session.setState("...");                             // encode once
 *   session.setMenu(["up", "down", "left", "right"]);    // encode once
 *   const r = session.score("Which move ...?");          // cheap
 *
 * Works unchanged in browsers and in Node >= 18. The module is built by
 * `make wasm` as a WASI reactor; the only imports it needs are the seven WASI
 * calls stubbed below, none of which touch a file system.
 *
 * Memory: the container is copied once into wasm memory and opened with
 * blink_model_open_memory, which borrows it. Each session owns one arena from
 * blink_session_create plus a scratch buffer for the text it passes in, so a
 * blink_score call allocates nothing on either side of the boundary once the
 * scratch buffer has reached its working size.
 *
 * SPDX-License-Identifier: Apache-2.0 */

const STATUS = [
  "ok", "invalid argument", "i/o error", "not a .blink container",
  "unsupported container version", "checksum mismatch", "arena too small",
  "input exceeds session limits", "state or menu not set",
];

export class BlinkError extends Error {
  constructor(where, status) {
    super(`${where}: ${STATUS[status] ?? `status ${status}`}`);
    this.status = status;
  }
}

function wasiShim(getMemory) {
  const EBADF = 8;
  const decoder = new TextDecoder();
  const now = typeof performance !== "undefined"
    ? () => performance.now() : () => Date.now();
  return {
    clock_time_get(_id, _precision, out) {
      const ns = BigInt(Math.round(now() * 1e6));
      new DataView(getMemory().buffer).setBigUint64(out, ns, true);
      return 0;
    },
    fd_write(fd, iovs, count, written) {
      const view = new DataView(getMemory().buffer);
      let text = "", total = 0;
      for (let i = 0; i < count; ++i) {
        const base = view.getUint32(iovs + i * 8, true);
        const length = view.getUint32(iovs + i * 8 + 4, true);
        text += decoder.decode(new Uint8Array(getMemory().buffer, base, length));
        total += length;
      }
      (fd === 2 ? console.error : console.log)(text.replace(/\n$/, ""));
      view.setUint32(written, total, true);
      return 0;
    },
    fd_close: () => EBADF,
    fd_seek: () => EBADF,
    fd_prestat_get: () => EBADF,        // no preopened directories
    fd_prestat_dir_name: () => EBADF,
    proc_exit(code) { throw new Error(`blink.wasm exited with ${code}`); },
  };
}

async function instantiate(source, imports) {
  if (source instanceof WebAssembly.Module) {
    return WebAssembly.instantiate(source, imports);
  }
  if (typeof source === "string" || source instanceof URL) {
    source = await fetch(source);
  }
  if (typeof Response !== "undefined" && source instanceof Response) {
    if (WebAssembly.instantiateStreaming) {
      try {
        return (await WebAssembly.instantiateStreaming(source.clone(), imports)).instance;
      } catch { /* wrong MIME type on a static server: fall through */ }
    }
    source = await source.arrayBuffer();
  }
  return (await WebAssembly.instantiate(source, imports)).instance;
}

export async function loadBlink(source) {
  let memory = null;
  const instance = await instantiate(source, {
    wasi_snapshot_preview1: wasiShim(() => memory),
  });
  const x = instance.exports;
  memory = x.memory;
  x._initialize();

  const encoder = new TextEncoder();
  const u8 = () => new Uint8Array(memory.buffer);
  const dv = () => new DataView(memory.buffer);
  const cstring = (ptr) => {
    const bytes = u8();
    let end = ptr;
    while (bytes[end]) ++end;
    return new TextDecoder().decode(bytes.subarray(ptr, end));
  };
  const malloc = (n) => {
    const p = x.malloc(Math.max(1, n));
    if (!p) throw new Error("blink.wasm: out of memory");
    return p;
  };

  /* One status word, reused by every call that reports through a pointer. */
  const statusPtr = malloc(4);
  const check = (where, status) => {
    if (status !== 0) throw new BlinkError(where, status);
  };

  class Session {
    constructor(model, limits) {
      this.model = model;
      const lim = malloc(16);
      const v = dv();
      v.setUint32(lim, limits.maxState ?? 0, true);
      v.setUint32(lim + 4, limits.maxQuestion ?? 0, true);
      v.setUint32(lim + 8, limits.maxOptions ?? 0, true);
      v.setUint32(lim + 12, limits.maxOption ?? 0, true);
      this.arenaBytes = x.blink_session_size(model.ptr, lim);
      this.ptr = x.blink_session_create(model.ptr, lim, statusPtr);
      x.free(lim);
      if (!this.ptr) throw new BlinkError("session", dv().getUint32(statusPtr, true));
      this.scratch = 0;
      this.scratchSize = 0;
      this.result = malloc(48);        /* sizeof(blink_result) on wasm32 */
      this.probs = malloc(64 * 4);     /* BLINK_MAX_OPTIONS floats       */
      this.menuSize = 0;
    }

    /* Grow-only scratch buffer for text crossing into wasm memory. */
    reserve(bytes) {
      if (bytes > this.scratchSize) {
        if (this.scratch) x.free(this.scratch);
        this.scratchSize = Math.max(bytes, this.scratchSize * 2, 1024);
        this.scratch = malloc(this.scratchSize);
      }
      return this.scratch;
    }

    writeText(text, at) {
      const bytes = encoder.encode(text);
      u8().set(bytes, at);
      return bytes.length;
    }

    setState(text) {
      const bytes = encoder.encode(text);
      const p = this.reserve(bytes.length);
      u8().set(bytes, p);
      check("state", x.blink_state_set(this.ptr, p, bytes.length));
    }

    setMenu(options) {
      const encoded = options.map((o) => encoder.encode(o));
      const n = encoded.length;
      const text = encoded.reduce((s, e) => s + e.length, 0);
      const p = this.reserve(n * 8 + text);
      const v = dv(), bytes = u8();
      let cursor = p + n * 8;
      encoded.forEach((e, i) => {
        v.setUint32(p + i * 4, cursor, true);          /* const char *[] */
        v.setUint32(p + n * 4 + i * 4, e.length, true); /* size_t[]       */
        bytes.set(e, cursor);
        cursor += e.length;
      });
      check("menu", x.blink_menu_set(this.ptr, p, p + n * 4, n));
      this.menuSize = n;
    }

    score(question) {
      const bytes = encoder.encode(question);
      const p = this.reserve(bytes.length);
      u8().set(bytes, p);
      check("score", x.blink_score(this.ptr, p, bytes.length, this.probs, this.result));
      return this.readResult();
    }

    readResult() {
      const v = dv();
      const r = this.result;
      const count = v.getUint32(r, true);
      return {
        probabilities: Array.from(new Float32Array(memory.buffer, this.probs, count)),
        argmax: v.getUint32(r + 4, true),
        confidence: v.getFloat32(r + 8, true),
        entropy: v.getFloat32(r + 12, true),
        margin: v.getFloat32(r + 16, true),
        stateBytes: v.getUint32(r + 20, true),
        questionBytes: v.getUint32(r + 24, true),
        contextPositions: v.getUint32(r + 28, true),
        encodeSeconds: v.getFloat64(r + 32, true),
        headSeconds: v.getFloat64(r + 40, true),
      };
    }

    /* Raw pre-temperature logits of the last score. */
    logits() {
      const countPtr = statusPtr;
      const p = x.blink_last_logits(this.ptr, countPtr);
      if (!p) return null;
      return Array.from(new Float32Array(memory.buffer, p, dv().getUint32(countPtr, true)));
    }

    free() {
      if (!this.ptr) return;
      x.blink_session_free(this.ptr);
      x.free(this.result);
      x.free(this.probs);
      if (this.scratch) x.free(this.scratch);
      this.ptr = 0;
    }
  }

  class Model {
    constructor(bytes, verify) {
      const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      /* The container is borrowed by the model, so it stays resident until
       * close(). 64-byte alignment matches what a page-aligned mmap gives the
       * native build, and the loader's own alignment checks require 16. */
      this.raw = malloc(data.length + 64);
      this.data = (this.raw + 63) & ~63;
      u8().set(data, this.data);
      this.ptr = x.blink_model_open_memory(this.data, data.length, verify ? 1 : 0, statusPtr);
      if (!this.ptr) {
        const status = dv().getUint32(statusPtr, true);
        x.free(this.raw);
        throw new BlinkError("model", status);
      }
    }

    info() {
      const p = malloc(128);
      x.blink_model_get_info(this.ptr, p);
      const v = dv();
      const u = (i) => v.getUint32(p + i * 4, true);
      const info = {
        width: u(0), blocks: u(1), ffnWidth: u(2), rank: u(3), heads: u(4),
        convWidth: u(5), stride: u(6), mixerBlocks: u(7), film: u(8),
        cross: u(9), bigramBuckets: u(10), maxState: u(11),
        maxQuestion: u(12), maxOption: u(13),
        temperature: v.getFloat32(p + 56, true),
        parameters: Number(v.getBigUint64(p + 64, true)),
        weightsBytes: Number(v.getBigUint64(p + 72, true)),
        name: cstring(p + 80),
      };
      x.free(p);
      return info;
    }

    createSession(limits = {}) { return new Session(this, limits); }

    close() {
      if (!this.ptr) return;
      x.blink_model_close(this.ptr);
      x.free(this.raw);
      this.ptr = 0;
    }
  }

  return {
    version: cstring(x.blink_version()),
    statusString: (s) => cstring(x.blink_status_string(s)),
    openModel: (bytes, { verify = true } = {}) => new Model(bytes, verify),
    memory,
  };
}
