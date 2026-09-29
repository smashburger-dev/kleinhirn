// Bisection probe for the WASM graph: runs each primitive on synthetic
// data and compares against a scalar JS reference. Node, no browser.
// Usage: node bench/dbg-wasm.mjs

import { readFileSync } from 'node:fs';

const wasm = readFileSync(new URL('../src/wasm/deberta.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm, {
  env: { abort: () => { throw new Error('abort'); } },
});
const ex = instance.exports;

const H = 384, HDIM = 384, I = 1536, HEADS = 6, D = 64, K_MAX = 16;
const wbase = ex.allocWeights(64 * 1024 * 1024);
ex.init(1024, 1e-7, 1.0);
const buf = () => ex.memory.buffer;
const f32 = (ptr, n) => new Float32Array(buf(), ptr, n);
const u32 = (ptr, n) => new Uint32Array(buf(), ptr, n);

let seed = 12345;
const rnd = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return (seed / 0x7fffffff - 0.5) * 2;
};

function report(name, got, want, tol = 1e-5) {
  let mx = 0;
  for (let i = 0; i < want.length; i++) mx = Math.max(mx, Math.abs(got[i] - want[i]));
  console.log(`${name}: maxDiff=${mx.toExponential(3)} ${mx <= tol ? 'OK' : 'FAIL'}`);
  return mx <= tol;
}

// --- matmul: C[M,N] = A[M,K] @ Wt[K,N] + B[N] ---
{
  const M = 5, K = 8, N = 12;
  const a = wbase, w = wbase + 4096, b = wbase + 8192, c = wbase + 12288;
  const A = [], W = [], B = [];
  for (let i = 0; i < M * K; i++) A.push(rnd());
  for (let i = 0; i < K * N; i++) W.push(rnd());
  for (let i = 0; i < N; i++) B.push(rnd());
  f32(a, M * K).set(A); f32(w, K * N).set(W); f32(b, N).set(B);
  ex.dbgMatmul(a, w, b, c, M, N, K, 0);
  const want = [];
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
    let s = B[n];
    for (let k = 0; k < K; k++) s += A[m * K + k] * W[k * N + n];
    want.push(s);
  }
  report('matmul-plain', f32(c, M * N), want);
  // relu
  ex.dbgMatmul(a, w, b, c, M, N, K, 1);
  report('matmul-relu', f32(c, M * N), want.map((v) => Math.max(v, 0)));
  // gelu vs JS erf same formula
  const g = (v) => {
    const u = v * 0.7071067811865476;
    const t = 1 / (1 + 0.3275911 * Math.abs(u));
    const p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t;
    const e = 1 - p * Math.exp(-u * u);
    return 0.5 * v * (1 + (v >= 0 ? e : -e));
  };
  ex.dbgMatmul(a, w, b, c, M, N, K, 2);
  report('matmul-gelu', f32(c, M * N), want.map(g));
}

// --- lnMasked: x[i] = (LN(emb[i])*w+b) * mask[i] over HIDDEN=384 ---
{
  const embP = ex.embPtr(), maskP = ex.maskPtr(), xP = ex.xPtr();
  const embLnW = wbase + (1 << 20), embLnB = wbase + (2 << 20);
  ex.setShared(embLnW, embLnB, 0, 0, 0, 0);
  const rows = 6;
  const E = [], Wln = [], Bln = [], M = [];
  for (let i = 0; i < rows * H; i++) E.push(rnd());
  for (let i = 0; i < H; i++) Wln.push(rnd()), Bln.push(rnd());
  for (let i = 0; i < rows; i++) M.push(i % 3 === 2 ? 0 : 1);
  f32(embP, 1024 * H).set(E.concat(new Array(1024 * H - E.length).fill(0)));
  f32(embLnW, H).set(Wln); f32(embLnB, H).set(Bln);
  f32(maskP, 1024).set(M.concat(new Array(1024 - M.length).fill(0)));
  ex.dbgLnMasked(rows);
  const want = [];
  for (let i = 0; i < rows; i++) {
    const row = E.slice(i * H, (i + 1) * H);
    const mean = row.reduce((s, v) => s + v, 0) / H;
    const sq = row.reduce((s, v) => s + v * v, 0) / H;
    const inv = 1 / Math.sqrt(sq - mean * mean + 1e-7);
    for (let d = 0; d < H; d++) {
      want.push(((row[d] - mean) * inv * Wln[d] + Bln[d]) * M[i]);
    }
  }
  report('lnMasked', f32(xP, rows * H), want, 1e-5);
}

// --- lnResidualW ---
{
  const a = wbase + (3 << 20), b = wbase + (4 << 20), o = wbase + (5 << 20);
  const wln = wbase + (6 << 20), bln = wbase + (7 << 20);
  const rows = 4;
  const A = [], B = [], Wln = [], Bln = [];
  for (let i = 0; i < rows * H; i++) A.push(rnd()), B.push(rnd());
  for (let i = 0; i < H; i++) Wln.push(rnd()), Bln.push(rnd());
  f32(a, rows * H).set(A); f32(b, rows * H).set(B);
  f32(wln, H).set(Wln); f32(bln, H).set(Bln);
  ex.dbgLnRes(a, b, o, wln, bln, rows);
  const want = [];
  for (let i = 0; i < rows; i++) {
    const row = A.slice(i * H, (i + 1) * H).map((v, d) => v + B[i * H + d]);
    const mean = row.reduce((s, v) => s + v, 0) / H;
    const sq = row.reduce((s, v) => s + v * v, 0) / H;
    const inv = 1 / Math.sqrt(sq - mean * mean + 1e-7);
    for (let d = 0; d < H; d++) want.push((row[d] - mean) * inv * Wln[d] + Bln[d]);
  }
  report('lnResidual', f32(o, rows * H), want, 1e-5);
}

// --- attention on synthetic qkv + pos tables, L=32 bucket of 1024 table ---
{
  const L = 32, rows = 6;
  const qkvP = ex.qkvPtr(), ctxP = ex.ctxPtr(), maskP = ex.maskPtr(), relP = ex.relidxPtr();
  const posK = wbase + (8 << 20), posQ = wbase + (8 << 20) + 512 * HDIM * 4;
  // register layer 0 pos tables (other pointers 0)
  ex.setLayer(0, 0, 0, posK, posQ, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  const QKV = [], PK = [], PQ = [], M = [], REL = [];
  for (let i = 0; i < L * 3 * HDIM; i++) QKV.push(rnd() * 0.2);
  for (let i = 0; i < 512 * HDIM; i++) PK.push(rnd() * 0.2), PQ.push(rnd() * 0.2);
  for (let i = 0; i < L; i++) M.push(i < rows ? 1 : 0);
  // relidx table at stride 1024: use simple delta-clamped indices 0..511
  for (let i = 0; i < 1024 * 1024; i++) REL.push(0);
  const rel = (i, j) => Math.min(Math.max(i - j + 256, 0), 511);
  for (let i = 0; i < 1024; i++) for (let j = 0; j < 1024; j++) REL[i * 1024 + j] = rel(i, j);
  f32(qkvP, L * 3 * HDIM).set(QKV);
  f32(posK, 512 * HDIM).set(PK); f32(posQ, 512 * HDIM).set(PQ);
  f32(maskP, 1024).set(M.concat(new Array(1024 - L).fill(0)));
  u32(relP, 1024 * 1024).set(REL);
  ex.dbgAttention(0, rows, L);
  // JS reference
  const SCALE = Math.sqrt(3 * D), hd = HDIM;
  const want = [];
  for (let i = 0; i < rows; i++) for (let h = 0; h < HEADS; h++) {
    const scores = [];
    for (let j = 0; j < L; j++) {
      const p = rel(i, j) * hd;
      let s = 0;
      for (let d = 0; d < D; d++) {
        const q = QKV[(i * 3 * hd) + h * D + d];
        const k = QKV[(j * 3 * hd) + (HEADS + h) * D + d];
        s += q * k + q * PK[p + h * D + d] + k * PQ[p + h * D + d];
      }
      scores.push(M[j] > 0.5 ? s / SCALE : -1e4);
    }
    const mx = Math.max(...scores);
    const exps = scores.map((s) => Math.exp(s - mx));
    const sum = exps.reduce((a2, b2) => a2 + b2, 0);
    for (let d = 0; d < D; d++) {
      let acc = 0;
      for (let j = 0; j < L; j++) {
        const sv = exps[j] / sum;
        if (sv !== 0) acc += sv * QKV[j * 3 * hd + (2 * HEADS + h) * D + d];
      }
      want.push(acc);
    }
  }
  const got = f32(ctxP, rows * HDIM);
  // reorder got [i][h*D+d] -> same layout
  report('attention', got, want, 1e-4);
}
console.log('probe done');
