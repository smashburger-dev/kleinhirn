// R8 matmul micro-bench core (docs/R8_WORKORDER.md Phase 0 step 3): runs the variants of
// src/wasm/gemm-probe.ts on one instance and returns GMAC/s per shape and variant, plus the error
// of every variant against an f64 reference and the multiply-add peak. Pure: the browser worker
// (bench/r8-gemm-worker.ts) and a Node check share it.

export interface ProbeExports {
  memory: WebAssembly.Memory;
  heapBase(): number;
  peak(n: number): number;
  [v: string]: unknown;
}

export interface Shape { M: number; K: number; N: number }

export const SHAPES: Shape[] = [
  { M: 128, K: 384, N: 1152 }, { M: 128, K: 384, N: 1536 }, { M: 128, K: 1536, N: 384 },
  { M: 128, K: 768, N: 2304 }, { M: 128, K: 768, N: 3072 }, { M: 128, K: 3072, N: 768 },
];

// variant -> layout of B it reads
export const VARIANTS: Record<string, 'kn' | 'p16' | 'p8'> = {
  v0: 'kn', v1: 'kn', v2: 'kn', v3: 'p16', v4: 'p8', v5: 'p16', v6: 'p16',
};

export interface ShapeResult extends Shape {
  gmacs: Record<string, number>;
  maxErr: Record<string, number>;
}

export interface ProbeResult { shapes: ShapeResult[]; peakGmacs: number }

export function runProbe(ex: ProbeExports, now: () => number, reps = 5, variants = Object.keys(VARIANTS), peakIters = 2e7): ProbeResult {
  const maxBytes = Math.max(...SHAPES.map((s) => 4 * (s.M * s.K + 3 * s.K * s.N + s.M * s.N))) + (1 << 16);
  let base = Math.ceil(ex.heapBase() / 64) * 64;
  const need = base + maxBytes - ex.memory.buffer.byteLength;
  if (need > 0) ex.memory.grow(Math.ceil(need / 65536));
  const shapes: ShapeResult[] = [];
  let seed = 1;
  const rnd = (): number => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 2 ** 32 - 0.5; };
  for (const s of SHAPES) {
    const { M, K, N } = s;
    let p = base;
    const al = (n: number): number => { const q = p; p += Math.ceil((4 * n) / 64) * 64 + 64; return q; };
    const A = al(M * K); const B = al(K * N); const P16 = al(K * N); const P8 = al(K * N); const C = al(M * N);
    const f = (q: number, n: number): Float32Array => new Float32Array(ex.memory.buffer, q, n);
    const a = f(A, M * K); const b = f(B, K * N);
    for (let i = 0; i < a.length; i += 1) a[i] = rnd();
    for (let i = 0; i < b.length; i += 1) b[i] = rnd();
    const p16 = f(P16, K * N); const p8 = f(P8, K * N);
    for (let k = 0; k < K; k += 1) {
      for (let n = 0; n < N; n += 1) {
        p16[(n >> 4) * K * 16 + k * 16 + (n & 15)] = b[k * N + n];
        p8[(n >> 3) * K * 8 + k * 8 + (n & 7)] = b[k * N + n];
      }
    }
    // f64 reference of the first 8 rows (enough to catch a wrong index)
    const R = 8;
    const ref = new Float64Array(R * N);
    for (let m = 0; m < R; m += 1) for (let k = 0; k < K; k += 1) { const x = a[m * K + k]; for (let n = 0; n < N; n += 1) ref[m * N + n] += x * b[k * N + n]; }
    const out: ShapeResult = { ...s, gmacs: {}, maxErr: {} };
    for (const v of variants) {
      const fn = ex[v] as (a: number, b: number, c: number, M: number, N: number, K: number) => void;
      const bp = VARIANTS[v] === 'kn' ? B : VARIANTS[v] === 'p16' ? P16 : P8;
      f(C, M * N).fill(Number.NaN);
      fn(A, bp, C, M, N, K);
      const c = f(C, M * N);
      let err = 0;
      for (let i = 0; i < R * N; i += 1) err = Math.max(err, Math.abs(c[i] - ref[i]));
      for (let i = R * N; i < M * N; i += 1) if (!Number.isFinite(c[i])) err = Infinity;
      const t: number[] = [];
      for (let r = 0; r < reps; r += 1) { const t0 = now(); fn(A, bp, C, M, N, K); t.push(now() - t0); }
      t.sort((x, y) => x - y);
      out.gmacs[v] = (M * N * K) / t[t.length >> 1] / 1e6;
      out.maxErr[v] = err;
    }
    shapes.push(out);
  }
  ex.peak(peakIters / 10);
  const t0 = now();
  ex.peak(peakIters);
  return { shapes, peakGmacs: (64 * peakIters) / (now() - t0) / 1e6 };
}
