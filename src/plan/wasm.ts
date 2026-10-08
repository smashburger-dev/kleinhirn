// Host side of the WASM executor (R2, docs/R2_WORKORDER.md Festlegungen 1, 3 to 5). One
// WebAssembly.Memory holds the weight arena and, per plan, the plan buffers at fixed addresses
// and the command stream src/wasm/plan.ts runs. A plan is translated once; a call writes the
// inputs, runs the stream with one export call and reads the output. f32 plans only. Pure: no
// Worker, no fetch, so the worker (src/wasm-plan-worker.ts) and the Node parity run share it.

import type { Op, Plan } from './ir.ts';
import type { RunInput } from './executor.ts';

const HEAD = 8;
const OPW = 32;
// Every allocation ends in this many spare bytes: the 4 x 16 tile loads 16 lanes at a column
// tail and discards the lanes past N.
const PAD = 64;
// Panel width of the weight matmul (R8 hc2): a weight with N >= 16 output columns is stored as
// [ceil(N / 16)][K][16], columns past N zero.
const NR = 16;
const PAGE = 65536;

const CODE = {
  gemmP: 1, gemmNT: 2, layernorm: 3, embln: 4, add: 5, gather: 6, pool: 7, geglu: 8, rope: 9,
  im2col: 10, attscore: 11, attsoftmax: 12, attpv: 13, attrel: 14, attsoftrel: 15, attstd: 16,
  attrelf: 17, masklogits: 18,
} as const;

const MATMUL = new Set(['matmul', 'mmtile', 'mmtile8', 'mmtile16']);

// Matmul weights with at least 16 output columns are stored in panels of 16 columns
// ([ceil(N / 16)][K][16]) so the 4 x 16 tile reads one contiguous 64-byte row per k; narrower ones
// stay [N, K] for dot products. The role of a tensor is the same in every plan of a model; a
// tensor in two roles is an error.
export function panelWeights(plans: Plan[]): Set<string> {
  const role = new Map<string, string>();
  const set = (name: string, r: string): void => {
    const was = role.get(name);
    if (was !== undefined && was !== r) throw new Error(`weight ${name} is bound as ${was} and as ${r}`);
    role.set(name, r);
  };
  for (const plan of plans) {
    for (const seg of plan.segments) {
      for (const op of seg.ops) {
        op.bind.forEach((id, i) => {
          if (!id.startsWith('w:')) return;
          const name = id.slice(2);
          set(name, MATMUL.has(op.kernel) && i === 1 ? (op.constants.N >= 16 ? 'kn' : 'nk') : 'plain');
        });
      }
    }
  }
  return new Set([...role].filter(([, r]) => r === 'kn').map(([n]) => n));
}

// R8 hc5: the split attention of the bucket plans as one fused op each, so a head's score block
// stays in the scratch block instead of the H L L scores buffer (12 MiB at L512, FINDINGS §55).
// Standard: attscore, attsoftmax, attpv -> the kernel of mbattention (attstd, same arithmetic).
// Relative (DeBERTa): attrel c2p, attrel p2c, attscore, attsoftrel, attpv -> the kernel of the
// fused attention (attrelf: (q.k + c2p + p2c) / scale in one rounding step instead of two).
export function fuseAttention(ops: Op[]): Op[] {
  const out: Op[] = [];
  const k = (i: number): string | undefined => ops[i]?.kernel;
  for (let i = 0; i < ops.length; i += 1) {
    const o = ops[i];
    if (o.kernel === 'attscore' && k(i + 1) === 'attsoftmax' && k(i + 2) === 'attpv'
      && ops[i + 1].bind[1] === o.bind[2] && ops[i + 2].bind[0] === o.bind[2]) {
      const c = o.constants;
      out.push({ name: `${o.name}+fused`, kernel: 'mbattention',
        constants: { L: c.L, H: c.H, D: c.D, SCALE: c.SCALE, WINDOW: ops[i + 1].constants.WINDOW },
        bind: [o.bind[0], ops[i + 1].bind[0], ops[i + 2].bind[4]], dispatch: o.dispatch });
      i += 2;
    } else if (o.kernel === 'attrel' && o.constants.PART === 0 && k(i + 1) === 'attrel' && ops[i + 1].constants.PART === 1
      && k(i + 2) === 'attscore' && k(i + 3) === 'attsoftrel' && k(i + 4) === 'attpv') {
      const s = ops[i + 2];
      out.push({ name: `${s.name}+fused`, kernel: 'attention',
        constants: { L: s.constants.L, H: s.constants.H, D: s.constants.D, SCALE: 1 / s.constants.SCALE },
        bind: [o.bind[0], o.bind[1], ops[i + 1].bind[1], o.bind[3], ops[i + 3].bind[0], ops[i + 4].bind[4]], dispatch: s.dispatch });
      i += 4;
    } else {
      out.push(o);
    }
  }
  return out;
}

// R8 hc6: the builds of src/wasm/plan.ts. 'relaxed' turns the matmul multiply-add into
// f32x4.relaxed_madd (a fused FMA in V8 and SpiderMonkey on ARM64); it validates in Chromium and
// Firefox only (FINDINGS §50). 'auto' takes relaxed where it validates.
export type WasmBuild = 'plain' | 'relaxed';

// One function: three v128.const and f32x4.relaxed_madd (the probe of bench/kbench.ts).
export function relaxedSimd(): boolean {
  const c = [0xfd, 0x0c, ...new Array<number>(16).fill(0)];
  const body = [0, ...c, ...c, ...c, 0xfd, 0x85, 0x02, 0x0b];
  return WebAssembly.validate(new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, 1, 5, 1, 0x60, 0, 1, 0x7b,
    3, 2, 1, 0, 10, body.length + 2, 1, body.length, ...body]));
}

export const resolveBuild = (b: WasmBuild | 'auto' | undefined): WasmBuild =>
  (b === 'plain' || b === 'relaxed' ? b : relaxedSimd() ? 'relaxed' : 'plain');

export interface WasmExports {
  memory: WebAssembly.Memory;
  heapBase(): number;
  run(stream: number, rows: number): void;
  runRange(stream: number, rows: number, from: number, to: number): void;
  helper(ctl: number, tid: number): void;
  wake(ctl: number): void;
}

// R8 Festlegung 6: the threaded build (plan-mt.wasm) imports a shared memory; helpers are workers
// that instantiate the same module on it and call helper(ctl, tid) until stop.
// ready resolves once the helper has instantiated the module, just before it enters helper():
// a browser hands a message to a worker started from a worker only while the parent's event
// loop runs, and the coordinator blocks in run() until every helper reports, so the host waits
// for all helpers at load (R8 hc7, the first browser run hung in warmup).
export type SpawnHelper = (module: WebAssembly.Module, memory: WebAssembly.Memory, ctl: number, tid: number) =>
  { ready: Promise<void>; terminate(): void };
const HELPER_START_MS = 10000;

// Words of the control block (src/wasm/plan.ts, threads section).
const CTL = { stop: 6, threads: 8, rb: 9, cb: 10, qb: 11, minWork: 13, words: 16 } as const;
// Work split (hc8 searches these): matmul items of rb rows x cb columns, attention items of qb
// query rows; a matmul under minWork multiply-adds runs on the coordinator alone.
export interface ThreadSplit { rb: number; cb: number; qb: number; minWork: number }
export const DEFAULT_SPLIT: ThreadSplit = { rb: 16, cb: 64, qb: 32, minWork: 1 << 16 };

// A shared memory for the threaded build; the largest maximum the engine grants, down to 1 GiB.
export function sharedMemory(): WebAssembly.Memory {
  for (let maximum = 65536; maximum >= 16384; maximum /= 2) {
    try { return new WebAssembly.Memory({ initial: 16, maximum, shared: true }); } catch { /* smaller */ }
  }
  throw new Error('no shared WebAssembly.Memory of 1 GiB or more');
}

// R8 op profile: mean ms per op over `reps` forwards, each op timed alone, and the mean of whole
// forwards for the check that the parts add up.
export interface OpProfile {
  ops: { kernel: string; name: string; M?: number; N?: number; K?: number }[];
  ms: number[];
  totalMs: number;
  reps: number;
}

export interface WasmTensor {
  name: string;
  shape: number[];
  data: Float32Array; // row-major as in the manifest
}

const align = (n: number): number => Math.ceil(n / 64) * 64;

export class WasmPlanRuntime {
  private top: number;
  private addr = new Map<string, number>();
  private panels = new Set<string>();
  weightBytes = 0;
  planBytes = 0;
  // threads (R8): control block address, thread count, helper workers
  ctl = 0;
  threads = 1;
  private helpers: { ready: Promise<void>; terminate(): void }[] = [];

  constructor(readonly ex: WasmExports, readonly module?: WebAssembly.Module) {
    this.top = align(ex.heapBase());
  }

  // The threaded build on a shared memory (R8 Festlegung 6); n - 1 helpers through spawn.
  static async createThreaded(source: BufferSource | WebAssembly.Module, n: number, spawn: SpawnHelper,
    split: ThreadSplit = DEFAULT_SPLIT): Promise<WasmPlanRuntime> {
    const module = source instanceof WebAssembly.Module ? source : await WebAssembly.compile(source);
    const memory = sharedMemory();
    const instance = await WebAssembly.instantiate(module, { env: { memory, abort: () => { throw new Error('wasm abort'); } } });
    const ex = instance.exports as unknown as WasmExports;
    const rt = new WasmPlanRuntime({ ...ex, memory }, module);
    rt.ctl = rt.alloc(4 * CTL.words);
    const w = new Int32Array(memory.buffer, rt.ctl, CTL.words);
    w.fill(0);
    w[CTL.threads] = n;
    w[CTL.rb] = split.rb;
    w[CTL.cb] = split.cb;
    w[CTL.qb] = split.qb;
    w[CTL.minWork] = split.minWork;
    rt.threads = n;
    for (let tid = 1; tid < n; tid += 1) rt.helpers.push(spawn(module, memory, rt.ctl, tid));
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([Promise.all(rt.helpers.map((h) => h.ready)), new Promise((_, rej) => {
        timer = setTimeout(() => rej(new Error(`helpers not ready after ${HELPER_START_MS} ms`)), HELPER_START_MS);
      })]);
    } catch (e) {
      rt.stopThreads();
      throw e;
    } finally {
      clearTimeout(timer);
    }
    return rt;
  }

  // Stops the helpers (they return from helper()) and ends their workers.
  stopThreads(): void {
    if (!this.ctl) return;
    Atomics.store(new Int32Array(this.ex.memory.buffer, this.ctl, CTL.words), CTL.stop, 1);
    this.ex.wake(this.ctl);
    for (const h of this.helpers) h.terminate();
    this.helpers = [];
  }

  static async create(source: BufferSource | WebAssembly.Module): Promise<WasmPlanRuntime> {
    const imports = { env: { abort: () => { throw new Error('wasm abort'); } } };
    const instance = source instanceof WebAssembly.Module
      ? await WebAssembly.instantiate(source, imports)
      : (await WebAssembly.instantiate(source, imports)).instance;
    return new WasmPlanRuntime(instance.exports as unknown as WasmExports);
  }

  get memoryBytes(): number {
    return this.ex.memory.buffer.byteLength;
  }

  alloc(bytes: number): number {
    const p = this.top;
    this.top = align(p + bytes + PAD);
    const need = this.top - this.ex.memory.buffer.byteLength;
    if (need > 0) this.ex.memory.grow(Math.ceil(need / PAGE));
    return p;
  }

  f32(p: number, n: number): Float32Array {
    return new Float32Array(this.ex.memory.buffer, p, n);
  }

  // One tensor into the arena, packed into panels of NR columns when `panel` is set.
  addWeight(t: WasmTensor, panel: boolean): void {
    if (this.addr.has(t.name)) throw new Error(`weight ${t.name} loaded twice`);
    let bytes = t.data.byteLength;
    if (panel) {
      const [N, K] = t.shape;
      if (t.shape.length !== 2 || N * K !== t.data.length) throw new Error(`weight ${t.name}: shape ${t.shape} is no matrix`);
      bytes = 4 * Math.ceil(N / NR) * NR * K;
    }
    const p = this.alloc(bytes);
    const dst = this.f32(p, bytes / 4);
    if (panel) {
      const [N, K] = t.shape;
      dst.fill(0);
      for (let n = 0; n < N; n += 1) {
        const base = Math.floor(n / NR) * K * NR + (n % NR);
        for (let k = 0; k < K; k += 1) dst[base + k * NR] = t.data[n * K + k];
      }
    } else {
      dst.set(t.data);
    }
    this.addr.set(`w:${t.name}`, p);
    if (panel) this.panels.add(`w:${t.name}`);
    this.weightBytes += align(bytes + PAD);
  }

  plan(plan: Plan): WasmPlan {
    if (plan.f16) throw new Error('the WASM executor runs f32 plans');
    const start = this.top;
    const p = new WasmPlan(this, plan);
    this.planBytes += this.top - start;
    return p;
  }

  weight(id: string): number | undefined {
    return this.addr.get(id);
  }

  // true: the weight is stored in panels (packed at load)
  paneled(id: string): boolean {
    return this.panels.has(id);
  }
}

export class WasmPlan {
  readonly length: number;
  readonly batch: number;
  readonly markers: number;
  private bufs = new Map<string, { p: number; bytes: number }>();
  private stream: number;
  private opList: Op[];

  constructor(private rt: WasmPlanRuntime, private plan: Plan) {
    this.length = plan.length;
    this.batch = plan.batch;
    this.markers = plan.markers;
    const ops = fuseAttention(plan.segments.flatMap((s) => s.ops));
    // buffers no op reads after the fusion (the scores of a fused attention) get no memory
    const used = new Set<string>([...ops.flatMap((o) => o.bind), plan.output.buffer,
      ...Object.values(plan.inputs).filter((v): v is string => typeof v === 'string'),
      ...(plan.rowSelect ? [plan.rowSelect.table, plan.rowSelect.dst] : [])]);
    for (const d of plan.buffers) {
      if (!used.has(d.id)) continue;
      const p = rt.alloc(d.bytes);
      this.bufs.set(d.id, { p, bytes: d.bytes });
      if (d.init) {
        new Uint8Array(rt.ex.memory.buffer, p, d.bytes)
          .set(new Uint8Array(d.init.buffer, d.init.byteOffset, d.init.byteLength));
      }
    }
    this.opList = ops;
    const L = plan.length;
    // fused attention: the score block (and c2p, p2c) plus panels of keys or position rows (hc11)
    const scratch = Math.max(0, ...ops.map((o) => (o.kernel === 'attention' ? 5 * L * L + (2 * L + 16) * o.constants.D
      : o.kernel === 'mbattention' || o.kernel === 'mbflash' ? L * L + (L + 16) * o.constants.D : 0)));
    const kinfo = rt.alloc(8 * plan.batch);
    // one scratch block per thread (header word 6: bytes per block)
    const scratchBytes = align(4 * scratch);
    const scratchP = scratch ? rt.alloc(scratchBytes * rt.threads) : 0;
    this.stream = rt.alloc(4 * (HEAD + OPW * ops.length));
    const words = new DataView(rt.ex.memory.buffer, this.stream, 4 * (HEAD + OPW * ops.length));
    [ops.length, L, plan.batch, this.at(plan.inputs.mask), kinfo, scratchP, scratchBytes, rt.ctl]
      .forEach((v, i) => words.setInt32(4 * i, v, true));
    ops.forEach((op, i) => this.encode(op, words, 4 * (HEAD + OPW * i)));
  }

  private at(id: string): number {
    const p = id.startsWith('w:') ? this.rt.weight(id) : this.bufs.get(id)?.p;
    if (p === undefined) throw new Error(`wasm plan: ${id.startsWith('w:') ? 'weight' : 'buffer'} ${id} missing`);
    return p;
  }

  private bytes(id: string): number {
    const b = this.bufs.get(id);
    if (!b) throw new Error(`wasm plan: buffer ${id} missing`);
    return b.bytes;
  }

  // One record: code, fixed rows, then the arguments; {f} marks an f32 argument.
  private encode(op: Op, w: DataView, o: number): void {
    const c = op.constants;
    op.bind.forEach((id, i) => {
      if (this.rt.paneled(id) && !(MATMUL.has(op.kernel) && i === 1)) {
        throw new Error(`wasm plan: ${op.name} reads the paneled weight ${id} as a plain tensor`);
      }
    });
    const b = (i: number): number => this.at(op.bind[i]);
    const fixed = (d: Op['dispatch'][number], n: number): number => (typeof d === 'number' ? n : -1);
    let rec: [number, number, ...(number | { f: number })[]];
    switch (op.kernel) {
      case 'matmul': case 'mmtile': case 'mmtile8': case 'mmtile16': {
        rec = [this.rt.paneled(op.bind[1]) ? CODE.gemmP : CODE.gemmNT, fixed(op.dispatch[1], c.M),
          b(0), b(1), b(2), b(3), c.N, c.K, c.ACT, c.M];
        break;
      }
      case 'layernorm':
        rec = [CODE.layernorm, fixed(op.dispatch[0], op.dispatch[0] as number),
          b(0), b(1), b(2), b(3), b(4), b(5), c.N, c.MODE, { f: c.EPS }];
        break;
      case 'embln':
        rec = [CODE.embln, -1, b(0), b(1), b(2), b(3), b(4), b(5), b(6), b(7),
          c.N, c.L, c.OFFSET, c.MAXPOS, { f: c.EPS }, c.MASKMUL, c.POSIDS];
        break;
      case 'add':
        rec = [CODE.add, -1, b(0), b(1), c.TOTAL, c.N, c.MODE, c.L];
        break;
      case 'gather':
        rec = [CODE.gather, -1, b(0), b(1), b(2), c.K, c.D, c.L, this.bytes(op.bind[0]) / 12];
        break;
      case 'pool':
        rec = [CODE.pool, -1, b(0), b(1), b(2), c.L, c.N, c.MODE, op.dispatch[1] as number];
        break;
      case 'geglu':
        rec = [CODE.geglu, -1, b(0), b(1), c.I];
        break;
      case 'rope':
        rec = [CODE.rope, -1, b(0), b(1), c.L, c.H, c.D];
        break;
      case 'im2col':
        rec = [CODE.im2col, -1, b(0), b(1), b(2), c.N, c.L, c.KS];
        break;
      case 'attscore':
        rec = [CODE.attscore, -1, b(0), b(2), c.L, c.H, c.D, { f: c.SCALE }];
        break;
      case 'attsoftmax':
        rec = [CODE.attsoftmax, -1, b(0), b(1), c.L, c.H, c.WINDOW];
        break;
      case 'attpv':
        rec = [CODE.attpv, -1, b(0), b(1), b(4), c.L, c.H, c.D];
        break;
      case 'attrel':
        rec = [CODE.attrel, -1, b(0), b(1), b(3), b(4), c.H, c.D, c.NM, c.MOFF, c.PART, c.L];
        break;
      case 'attsoftrel':
        rec = [CODE.attsoftrel, -1, b(0), b(1), b(2), b(3), b(4), c.L, c.H, c.NM, c.MOFF, { f: c.INVSCALE }];
        break;
      case 'mbattention': case 'mbflash':
        rec = [CODE.attstd, -1, b(0), b(1), b(2), c.L, c.H, c.D, { f: c.SCALE }, c.WINDOW];
        break;
      case 'attention':
        rec = [CODE.attrelf, -1, b(0), b(1), b(2), b(3), b(4), b(5), c.L, c.H, c.D, { f: c.SCALE }];
        break;
      case 'masklogits':
        rec = [CODE.masklogits, -1, b(0), b(1), b(2), c.K, { f: c.TEMP }, this.bytes(op.bind[1]) / 12];
        break;
      default:
        throw new Error(`wasm plan: kernel ${op.kernel as string} has no computation`);
    }
    rec.forEach((v, i) => {
      if (typeof v === 'number') {
        if (!Number.isInteger(v)) throw new Error(`wasm plan: ${op.name} argument ${i} is ${v}`);
        w.setInt32(o + 4 * i, v, true);
      } else {
        w.setFloat32(o + 4 * i, v.f, true);
      }
    });
  }

  upload(input: RunInput): void {
    const i = this.plan.inputs;
    const mem = this.rt.ex.memory.buffer;
    const put = (id: string, src: ArrayBufferView): void => {
      const b = this.bufs.get(id);
      if (!b) throw new Error(`wasm plan: input ${id} missing`);
      if (src.byteLength > b.bytes) throw new Error(`wasm plan: input ${id} has ${src.byteLength} B, the buffer ${b.bytes}`);
      new Uint8Array(mem, b.p, src.byteLength).set(new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
    };
    if (!(input.embeddings instanceof Float32Array)) throw new Error('the WASM executor takes f32 word rows');
    put(i.embeddings, input.embeddings);
    put(i.mask, input.mask);
    if (i.markers) {
      if (!input.packedMarkers) throw new Error('plan needs packedMarkers');
      put(i.markers, input.packedMarkers);
    }
    if (i.typeIds) {
      if (!input.typeIds) throw new Error('plan needs typeIds');
      put(i.typeIds, input.typeIds);
    }
    const rs = this.plan.rowSelect;
    if (rs) {
      // Julia: the type row of each sequence, as the copies before the first WebGPU pass
      const qtypes = Array.isArray(input.qtype) ? input.qtype : [input.qtype ?? 0];
      const table = this.at(rs.table);
      const dst = this.at(rs.dst);
      const u8 = new Uint8Array(this.rt.ex.memory.buffer);
      for (let b = 0; b < this.batch; b += 1) {
        u8.copyWithin(dst + b * rs.rowBytes, table + (qtypes[b] ?? 0) * rs.rowBytes, table + ((qtypes[b] ?? 0) + 1) * rs.rowBytes);
      }
    }
  }

  // rows: the real rows of a B = 1 call, length * batch for a batch plan.
  run(rows: number): void {
    if (rows < 1 || rows > this.length * this.batch) throw new Error(`wasm plan: ${rows} rows, the plan has ${this.length * this.batch}`);
    this.rt.ex.run(this.stream, rows);
  }

  // R8 diagnosis: each op timed alone with `now` (performance.now in the worker), `reps` forwards
  // after the inputs of the last upload; then `reps` whole forwards. The output is that of a whole
  // forward.
  profile(rows: number, reps: number, now: () => number): OpProfile {
    if (rows < 1 || rows > this.length * this.batch) throw new Error(`wasm plan: ${rows} rows, the plan has ${this.length * this.batch}`);
    const n = this.opList.length;
    const ms = new Array<number>(n).fill(0);
    for (let r = 0; r < reps; r += 1) {
      for (let o = 0; o < n; o += 1) {
        const t0 = now();
        this.rt.ex.runRange(this.stream, rows, o, o + 1);
        ms[o] += now() - t0;
      }
    }
    let total = 0;
    for (let r = 0; r < reps; r += 1) {
      const t0 = now();
      this.rt.ex.run(this.stream, rows);
      total += now() - t0;
    }
    const mm = (op: Op) => (MATMUL.has(op.kernel) ? { M: op.constants.M, N: op.constants.N, K: op.constants.K } : {});
    return { ops: this.opList.map((op) => ({ kernel: op.kernel, name: op.name, ...mm(op) })),
      ms: ms.map((v) => v / reps), totalMs: total / reps, reps };
  }

  // A plan buffer as f32 (tests, debugging); the view dies when the memory grows.
  view(id: string): Float32Array {
    return this.rt.f32(this.at(id), this.bytes(id) / 4);
  }

  // The output as a fresh f32 array of rows * cols values.
  readOutput(): Float32Array {
    const o = this.plan.output;
    return this.rt.f32(this.at(o.buffer), o.rows * o.cols).slice();
  }
}
