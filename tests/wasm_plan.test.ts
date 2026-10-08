// R2 stage 3 step 1 (docs/R2_WORKORDER.md): every computation of the WASM executor
// (src/wasm/plan.ts through src/plan/wasm.ts) against a JS computation in f64 on random inputs,
// with L and D outside groups of 4, masked rows and batches of two sequences.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import type { Op, Plan } from '../src/plan/ir.ts';
import { WasmPlanRuntime, type WasmPlan } from '../src/plan/wasm.ts';

const WASM = readFileSync(new URL('../src/wasm/plan.wasm', import.meta.url));

let seed = 12345;
const rand = (): number => {
  seed = (seed * 1103515245 + 12345) >>> 0;
  return seed / 4294967296 * 2 - 1;
};
const randArr = (n: number, s = 1): Float32Array => Float32Array.from({ length: n }, () => rand() * s);

// erf of GELU as in the kernels (Abramowitz-Stegun 7.1.26), in f64
const gelu = (v: number): number => {
  const u = v * Math.SQRT1_2;
  const t = 1 / (1 + 0.3275911 * Math.abs(u));
  const p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const e = 1 - p * Math.exp(-u * u);
  return 0.5 * v * (1 + (v >= 0 ? e : -e));
};
const ACT = [(v: number) => v, (v: number) => Math.max(v, 0), gelu, Math.tanh, (v: number) => v / (1 + Math.exp(-v))];

function close(got: ArrayLike<number>, want: ArrayLike<number>, what: string, tol = 2e-5): void {
  assert.equal(got.length, want.length, `${what}: length`);
  let scale = 1;
  for (let i = 0; i < want.length; i += 1) scale = Math.max(scale, Math.abs(want[i]));
  for (let i = 0; i < want.length; i += 1) {
    assert.ok(Number.isFinite(got[i]) && Math.abs(got[i] - want[i]) <= tol * scale,
      `${what}[${i}]: ${got[i]} vs ${want[i]}`);
  }
}

interface Mini {
  buffers: Record<string, number | Uint32Array | Float32Array>; // elements, or the init data
  ops: Op[];
  weights?: Record<string, { shape: number[]; data: Float32Array; panel?: boolean }>;
  L: number;
  batch?: number;
  out: string;
}

// A runtime with the weights and one plan of the given ops; the mask buffer is 'mask'.
async function mini(m: Mini): Promise<{ plan: WasmPlan; rt: WasmPlanRuntime }> {
  const rt = await WasmPlanRuntime.create(WASM);
  for (const [name, w] of Object.entries(m.weights ?? {})) rt.addWeight({ name, shape: w.shape, data: w.data }, !!w.panel);
  const buffers = Object.entries(m.buffers).map(([id, v]) => (typeof v === 'number'
    ? { id, bytes: 4 * v, usage: 'rw' as const }
    : { id, bytes: v.byteLength, usage: 'rw' as const, init: v }));
  const out = m.buffers[m.out] as number;
  const plan: Plan = {
    f16: false, length: m.L, batch: m.batch ?? 1, markers: 0, embeddingSize: 1, buffers,
    segments: [{ name: 's', prefix: '', ops: m.ops }],
    inputs: { embeddings: 'mask', mask: 'mask' },
    output: { buffer: m.out, bytes: 4 * out, dtype: 'f32', rows: out, cols: 1 },
    captureSlots: 0, captureSlotBytes: 0,
  };
  return { plan: rt.plan(plan), rt };
}

const prefixMask = (L: number, lens: number[]): Float32Array => {
  const m = new Float32Array(L * lens.length);
  lens.forEach((n, b) => m.fill(1, b * L, b * L + n));
  return m;
};

for (const [N, panel] of [[37, true], [3, false], [16, true], [40, true]] as const) {
  test(`matmul N=${N} (${panel ? 'panels' : 'dot'}) against f64, all activations`, async () => {
    const M = N === 40 ? 9 : 7; const K = 19;
    const W = randArr(N * K); const B = randArr(N);
    for (let act = 0; act < 5; act += 1) {
      for (const fixed of [false, true]) {
        const { plan } = await mini({
          L: M, out: 'c', buffers: { mask: M, a: M * K, c: M * N },
          weights: { w: { shape: [N, K], data: W, panel }, b: { shape: [N], data: B } },
          ops: [{ name: 'mm', kernel: 'mmtile', constants: { M, N, K, ACT: act }, bind: ['a', 'w:w', 'w:b', 'c'],
            dispatch: [1, fixed ? 1 : 'rows32'] }],
        });
        const A = randArr(M * K, 2);
        plan.view('a').set(A);
        const rows = fixed ? M : 5;
        plan.run(fixed ? 1 : rows);
        const want: number[] = [];
        for (let m = 0; m < rows; m += 1) {
          for (let n = 0; n < N; n += 1) {
            let s = 0;
            for (let k = 0; k < K; k += 1) s += A[m * K + k] * W[n * K + k];
            want.push(ACT[act](s + B[n]));
          }
        }
        close(plan.view('c').subarray(0, rows * N), want, `act ${act} fixed ${fixed}`);
      }
    }
  });
}

test('layernorm modes 0, 1, 2 (two-pass variance) and embln with positions, types, POSIDS', async () => {
  const N = 21; const rows = 5;
  const w = randArr(N); const bias = randArr(N);
  const ref = (v: number[], scale: number): number[] => {
    const mean = v.reduce((a, x) => a + x, 0) / N;
    const variance = v.reduce((a, x) => a + (x - mean) ** 2, 0) / N;
    const inv = 1 / Math.sqrt(variance + 1e-5);
    return v.map((x, d) => ((x - mean) * inv * w[d] + bias[d]) * scale);
  };
  for (const mode of [0, 1, 2]) {
    const { plan } = await mini({
      L: rows, out: 'o', buffers: { mask: rows, a: rows * N, b: rows * N, o: rows * N },
      weights: { w: { shape: [N], data: w }, bias: { shape: [N], data: bias } },
      ops: [{ name: 'ln', kernel: 'layernorm', constants: { N, MODE: mode, EPS: 1e-5 },
        bind: ['a', 'b', 'w:w', 'w:bias', 'mask', 'o'], dispatch: ['rows', 1] }],
    });
    const a = randArr(rows * N, 3); const b = randArr(rows * N, 3);
    const mask = Float32Array.from([1, 1, 0, 1, 0]);
    plan.view('a').set(a); plan.view('b').set(b); plan.view('mask').set(mask);
    plan.run(rows);
    const want: number[] = [];
    for (let r = 0; r < rows; r += 1) {
      const v = [...a.subarray(r * N, r * N + N)].map((x, d) => x + (mode === 2 ? b[r * N + d] : 0));
      want.push(...ref(v, mode === 1 ? mask[r] : 1));
    }
    close(plan.view('o'), want, `mode ${mode}`);
  }
  // embln: L 6, offset 2, 9 position rows, 3 types; POSIDS reads the row from the high 16 bits
  const L = 6; const P = 9;
  const pos = randArr(P * N); const typ = randArr(3 * N);
  for (const posids of [0, 1]) {
    for (const maskmul of [0, 1]) {
      const { plan } = await mini({
        L, batch: 2, out: 'o', buffers: { mask: 2 * L, word: 2 * L * N, tt: 2 * L, o: 2 * L * N },
        weights: { pos: { shape: [P, N], data: pos }, typ: { shape: [3, N], data: typ },
          w: { shape: [N], data: w }, bias: { shape: [N], data: bias } },
        ops: [{ name: 'e', kernel: 'embln', constants: { N, L, OFFSET: 2, MAXPOS: P, EPS: 1e-5, MASKMUL: maskmul, POSIDS: posids },
          bind: ['word', 'w:pos', 'w:typ', 'tt', 'w:w', 'w:bias', 'mask', 'o'], dispatch: ['rows', 1] }],
      });
      const word = randArr(2 * L * N);
      const types = Uint32Array.from({ length: 2 * L }, (_, i) => (i % 3) | (posids ? ((i * 5) % P) << 16 : 0));
      const mask = prefixMask(L, [4, 6]);
      plan.view('word').set(word); plan.view('mask').set(mask);
      new Uint32Array(plan.view('tt').buffer, plan.view('tt').byteOffset, 2 * L).set(types);
      plan.run(2 * L);
      const want: number[] = [];
      for (let r = 0; r < 2 * L; r += 1) {
        const prow = Math.min(posids ? types[r] >> 16 : (r % L) + 2, P - 1);
        const v = Array.from({ length: N }, (_, d) => word[r * N + d] + pos[prow * N + d] + typ[(types[r] & 0xffff) * N + d]);
        want.push(...ref(v, maskmul ? mask[r] : 1));
      }
      close(plan.view('o'), want, `embln posids ${posids} maskmul ${maskmul}`);
    }
  }
});

test('add, gather, pool, geglu, rope, im2col, masklogits', async () => {
  const L = 7; const N = 6; const B = 2;
  // add MODE 0 and 1 (one source row per sequence)
  for (const mode of [0, 1]) {
    const { plan } = await mini({
      L, batch: B, out: 'dst', buffers: { mask: B * L, dst: B * L * N, src: B * L * N },
      ops: [{ name: 'a', kernel: 'add', constants: { TOTAL: B * L * N, N, MODE: mode, L }, bind: ['dst', 'src'], dispatch: [1, 1] }],
    });
    const d = randArr(B * L * N); const s = randArr(B * L * N);
    plan.view('dst').set(d); plan.view('src').set(s);
    plan.run(B * L);
    close(plan.view('dst'), [...d].map((x, i) => x + (mode ? s[Math.floor(i / (N * L)) * N + i % N] : s[i])), `add ${mode}`);
  }
  // gather: K 3 markers per sequence
  {
    const K = 3;
    const marks = Uint32Array.from([2, 0, 5, 0, 0, 0, 0, 0, 0, 6, 1, 3, 0, 0, 0, 0, 0, 0]);
    const { plan } = await mini({
      L, batch: B, out: 'st', buffers: { mask: B * L, mk: marks, x: B * L * N, st: B * K * N },
      ops: [{ name: 'g', kernel: 'gather', constants: { K, D: N, L }, bind: ['mk', 'x', 'st'], dispatch: [1, 1] }],
    });
    const x = randArr(B * L * N);
    plan.view('x').set(x);
    plan.run(B * L);
    const want: number[] = [];
    for (let g = 0; g < B * K; g += 1) {
      const b = Math.floor(g / K);
      const row = b * L + marks[b * 3 * K + g - b * K];
      want.push(...x.subarray(row * N, row * N + N));
    }
    close(plan.view('st'), want, 'gather');
  }
  // pool mean and max, masks with padding rows holding NaN (never read)
  for (const mode of [0, 1]) {
    const { plan } = await mini({
      L, batch: B, out: 'o', buffers: { mask: B * L, x: B * L * N, o: B * N },
      ops: [{ name: 'p', kernel: 'pool', constants: { L, N, MODE: mode }, bind: ['x', 'mask', 'o'], dispatch: [1, B] }],
    });
    const x = randArr(B * L * N);
    const mask = prefixMask(L, [3, 7]);
    for (let r = 0; r < B * L; r += 1) if (!mask[r]) x.fill(NaN, r * N, r * N + N);
    plan.view('x').set(x); plan.view('mask').set(mask);
    plan.run(B * L);
    const want: number[] = [];
    for (let b = 0; b < B; b += 1) {
      for (let d = 0; d < N; d += 1) {
        const vals = [...Array(L).keys()].filter((i) => mask[b * L + i]).map((i) => x[(b * L + i) * N + d]);
        want.push(mode ? Math.max(...vals) : vals.reduce((a, v) => a + v, 0) / vals.length);
      }
    }
    close(plan.view('o'), want, `pool ${mode}`);
  }
  // geglu
  {
    const I = 5;
    const { plan } = await mini({
      L, out: 'g', buffers: { mask: L, mid: L * 2 * I, g: L * I },
      ops: [{ name: 'gg', kernel: 'geglu', constants: { I }, bind: ['mid', 'g'], dispatch: ['rows', 1] }],
    });
    const mid = randArr(L * 2 * I, 3);
    plan.view('mid').set(mid);
    plan.run(L);
    const want: number[] = [];
    for (let r = 0; r < L; r += 1) for (let j = 0; j < I; j += 1) want.push(gelu(mid[r * 2 * I + j]) * mid[r * 2 * I + I + j]);
    close(plan.view('g'), want, 'geglu');
  }
  // rope on q and k, H 2, D 6, two sequences
  {
    const H = 2; const D = 6;
    const table = new Float32Array(L * 2 * D);
    for (let i = 0; i < L; i += 1) {
      for (let d = 0; d < D / 2; d += 1) {
        const f = i * 10000 ** (-(2 * d) / D);
        table[i * 2 * D + d] = Math.cos(f); table[i * 2 * D + D + d] = Math.sin(f);
      }
    }
    const { plan } = await mini({
      L, batch: B, out: 'qkv', buffers: { mask: B * L, qkv: B * L * 3 * H * D, cs: table },
      ops: [{ name: 'r', kernel: 'rope', constants: { L, H, D }, bind: ['qkv', 'cs'], dispatch: ['rows', 2 * H] }],
    });
    const qkv = randArr(B * L * 3 * H * D);
    plan.view('qkv').set(qkv);
    plan.run(B * L);
    const want = [...qkv];
    for (let i = 0; i < B * L; i += 1) {
      const il = i % L;
      for (let s = 0; s < 2 * H; s += 1) {
        const base = i * 3 * H * D + s * D;
        for (let d = 0; d < D / 2; d += 1) {
          const x = qkv[base + d]; const y = qkv[base + d + D / 2];
          const c = table[il * 2 * D + d]; const sn = table[il * 2 * D + D + d];
          want[base + d] = x * c - y * sn; want[base + d + D / 2] = y * c + x * sn;
        }
      }
    }
    close(plan.view('qkv'), want, 'rope');
  }
  // im2col, KS 3, a padding row holding NaN reads as zero
  {
    const KS = 3;
    const { plan } = await mini({
      L, batch: B, out: 'o', buffers: { mask: B * L, x: B * L * N, o: B * L * KS * N },
      ops: [{ name: 'i', kernel: 'im2col', constants: { N, L, KS }, bind: ['x', 'mask', 'o'], dispatch: ['rows', 1] }],
    });
    const x = randArr(B * L * N);
    const mask = prefixMask(L, [5, 7]);
    for (let r = 0; r < B * L; r += 1) if (!mask[r]) x.fill(NaN, r * N, r * N + N);
    plan.view('x').set(x); plan.view('mask').set(mask);
    plan.run(B * L);
    const want: number[] = [];
    for (let r = 0; r < B * L; r += 1) {
      for (let t = 0; t < KS; t += 1) {
        const j = (r % L) + t - 1;
        const src = Math.floor(r / L) * L + Math.max(j, 0);
        const ok = j >= 0 && j < L && mask[src] !== 0;
        for (let c = 0; c < N; c += 1) want.push(ok ? x[src * N + c] : 0);
      }
    }
    close(plan.view('o'), want, 'im2col');
  }
  // masklogits: K 4 per sequence, mask bits as f32
  {
    const K = 4;
    const packed = new Uint32Array(3 * K * B);
    const mbits = new Float32Array(packed.buffer);
    [1, 0, 1, 1].forEach((v, k) => { mbits[K + k] = v; });
    [0, 1, 1, 0].forEach((v, k) => { mbits[3 * K + K + k] = v; });
    const { plan } = await mini({
      L, batch: B, out: 'lg', buffers: { mask: B * L, raw: B * K, pk: packed, lg: B * K },
      ops: [{ name: 'm', kernel: 'masklogits', constants: { K, TEMP: 0.5 }, bind: ['raw', 'pk', 'lg'], dispatch: [1, 1] }],
    });
    const raw = randArr(B * K);
    plan.view('raw').set(raw);
    plan.run(B * L);
    close(plan.view('lg'), [...raw].map((v, k) => (mbits[Math.floor(k / K) * 3 * K + K + k % K] > 0.5 ? v / 0.5 : -1e4)), 'masklogits');
  }
});

// Reference attention over packed sequences. rel: relative terms (DeBERTa); scale multiplies
// (standard) or divides (relative) like the kernels.
function attentionRef(
  qkv: Float32Array, mask: Float32Array, L: number, B: number, H: number, D: number, scale: number,
  window: number, rows: number, rel?: { idx: Uint32Array; pk: Float32Array; pq: Float32Array },
): number[] {
  const hd = H * D;
  const out = new Array<number>(rows * hd).fill(0);
  for (let i = 0; i < rows; i += 1) {
    const b = Math.floor(i / L); const il = i - b * L;
    if (mask[i] <= 0.5) continue;
    for (let h = 0; h < H; h += 1) {
      const keys: number[] = []; const s: number[] = [];
      for (let j = 0; j < L; j += 1) {
        if (mask[b * L + j] <= 0.5 || (window && Math.abs(il - j) > window)) continue;
        let v = 0;
        for (let d = 0; d < D; d += 1) {
          const q = qkv[i * 3 * hd + h * D + d]; const k = qkv[(b * L + j) * 3 * hd + (H + h) * D + d];
          v += q * k;
          if (rel) {
            const p = rel.idx[il * L + j];
            v += q * rel.pk[p * hd + h * D + d] + k * rel.pq[p * hd + h * D + d];
          }
        }
        keys.push(j); s.push(rel ? v / scale : v * scale);
      }
      const mx = Math.max(...s);
      const e = s.map((x) => Math.exp(x - mx));
      const sum = e.reduce((a, x) => a + x, 0);
      for (let d = 0; d < D; d += 1) {
        let acc = 0;
        keys.forEach((j, n) => { acc += (e[n] / sum) * qkv[(b * L + j) * 3 * hd + (2 * H + h) * D + d]; });
        out[i * hd + h * D + d] = acc;
      }
    }
  }
  return out;
}

const SPAN = 4;
const relTable = (L: number): Uint32Array => {
  const t = new Uint32Array(L * L);
  for (let i = 0; i < L; i += 1) for (let j = 0; j < L; j += 1) t[i * L + j] = Math.min(Math.max(i - j + SPAN, 0), 2 * SPAN - 1);
  return t;
};

for (const [L, D, lens, window] of [[7, 5, [5], 0], [9, 6, [4, 9], 0], [9, 6, [9, 6], 2], [12, 8, [12], 0]] as const) {
  const B = lens.length; const H = 2; const hd = H * D;
  const rows = B === 1 ? lens[0] : B * L;
  test(`standard attention L=${L} D=${D} lens=${lens} window=${window}: split and fused`, async () => {
    const qkv = randArr(B * L * 3 * hd, 2);
    const mask = prefixMask(L, [...lens]);
    // stale padding rows (NaN) must never enter a sum
    for (let r = 0; r < B * L; r += 1) if (!mask[r]) qkv.fill(NaN, r * 3 * hd, (r + 1) * 3 * hd);
    const want = attentionRef(qkv, mask, L, B, H, D, 0.3, window, rows);
    const base = { L, H, D };
    for (const fused of ['split', 'mbflash', 'mbattention']) {
      const ops: Op[] = fused === 'split' ? [
        { name: 's', kernel: 'attscore', constants: { ...base, SCALE: 0.3, ROWS: B * L }, bind: ['qkv', 'ki', 'sc'], dispatch: [1, 'rows32'] },
        { name: 'p', kernel: 'attsoftmax', constants: { L, H, WINDOW: window }, bind: ['mask', 'sc'], dispatch: [H, 'rows'] },
        { name: 'v', kernel: 'attpv', constants: { ...base, ROWS: B * L }, bind: ['sc', 'qkv', 'mask', 'ki', 'ctx'], dispatch: [1, 'rows32'] },
      ] : [{ name: 'a', kernel: fused as 'mbflash', constants: { ...base, SCALE: 0.3, WINDOW: window, ROWS: B * L }, bind: ['qkv', 'mask', 'ctx'], dispatch: [H, 'rows'] }];
      const { plan } = await mini({ L, batch: B, out: 'ctx', buffers: { mask: B * L, qkv: B * L * 3 * hd, ki: 2 * B, sc: B * L * H * L, ctx: B * L * hd }, ops });
      plan.view('qkv').set(qkv); plan.view('mask').set(mask);
      plan.run(rows);
      const got = plan.view('ctx').subarray(0, rows * hd);
      close(got.map((v, i) => (mask[Math.floor(i / hd)] ? v : 0)), want, fused);
      for (let i = 0; i < rows; i += 1) if (!mask[i]) assert.ok(got.subarray(i * hd, (i + 1) * hd).every((v) => v === 0), `${fused}: masked row ${i} not zero`);
    }
  });

  if (window) continue;
  test(`relative attention L=${L} D=${D} lens=${lens}: split and fused`, async () => {
    const qkv = randArr(B * L * 3 * hd, 2);
    const mask = prefixMask(L, [...lens]);
    for (let r = 0; r < B * L; r += 1) if (!mask[r]) qkv.fill(NaN, r * 3 * hd, (r + 1) * 3 * hd);
    const idx = relTable(L);
    const pk = randArr(2 * SPAN * hd); const pq = randArr(2 * SPAN * hd);
    const scale = Math.sqrt(3 * D);
    const want = attentionRef(qkv, mask, L, B, H, D, scale, 0, rows, { idx, pk, pq });
    const moff = idx[L - 1]; const nm = idx[(L - 1) * L] - moff + 1;
    const relC = { H, D, ROWS: B * L, NM: nm, MOFF: moff, L };
    for (const fused of [false, true]) {
      const ops: Op[] = fused
        ? [{ name: 'a', kernel: 'attention', constants: { L, H, D, SCALE: scale }, bind: ['qkv', 'w:pk', 'w:pq', 'ri', 'mask', 'ctx'], dispatch: [H, 'rows'] }]
        : [
          { name: 'c', kernel: 'attrel', constants: { ...relC, PART: 0 }, bind: ['qkv', 'w:pk', 'ki', 'ri', 'c2p'], dispatch: [1, 'rows32'] },
          { name: 'p', kernel: 'attrel', constants: { ...relC, PART: 1 }, bind: ['qkv', 'w:pq', 'ki', 'ri', 'p2c'], dispatch: [1, 'rows32'] },
          { name: 's', kernel: 'attscore', constants: { L, H, D, SCALE: 1 / scale, ROWS: B * L }, bind: ['qkv', 'ki', 'sc'], dispatch: [1, 'rows32'] },
          { name: 'sr', kernel: 'attsoftrel', constants: { L, H, NM: nm, MOFF: moff, INVSCALE: 1 / scale }, bind: ['mask', 'ri', 'c2p', 'p2c', 'sc'], dispatch: [H, 'rows'] },
          { name: 'v', kernel: 'attpv', constants: { L, H, D, ROWS: B * L }, bind: ['sc', 'qkv', 'mask', 'ki', 'ctx'], dispatch: [1, 'rows32'] },
        ];
      const { plan } = await mini({
        L, batch: B, out: 'ctx', ops,
        buffers: { mask: B * L, qkv: B * L * 3 * hd, ki: 2 * B, ri: idx, sc: B * L * H * L, c2p: B * L * H * nm, p2c: B * L * H * nm, ctx: B * L * hd },
        weights: { pk: { shape: [2 * SPAN, hd], data: pk }, pq: { shape: [2 * SPAN, hd], data: pq } },
      });
      plan.view('qkv').set(qkv); plan.view('mask').set(mask);
      plan.run(rows);
      const got = plan.view('ctx').subarray(0, rows * hd);
      close(got.map((v, i) => (mask[Math.floor(i / hd)] ? v : 0)), want, fused ? 'fused' : 'split');
      for (let i = 0; i < rows; i += 1) if (!mask[i]) assert.ok(got.subarray(i * hd, (i + 1) * hd).every((v) => v === 0), `masked row ${i} not zero`);
    }
  });
}
