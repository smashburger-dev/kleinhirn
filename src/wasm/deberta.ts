// AssemblyScript port of the DeBERTa-v2 + GLiNER2 head for browsers
// without WebGPU (docs/PLAN.md K5, fallback chain step 3).
// Same math as src/plan/layers.ts and the WGSL kernels, f32 throughout.
// Weight tensors live in one arena the host fills by byte offset; input
// regions (embeddings, mask, packed markers, relidx, logits) are allocated
// here and shared with JS through exported pointers.

const HIDDEN: i32 = 384;
const HEADS: i32 = 6;
const HEAD_DIM: i32 = 64;
const INTERMEDIATE: i32 = 1536;
const LAYERS: i32 = 12;
const K_MAX: i32 = 16;
const HEAD_HID: i32 = 768;
const HD: i32 = HEADS * HEAD_DIM;

let BUCKET: i32 = 0;
let EPS: f32 = 1e-7;
let TEMP: f32 = 1.0;

// ---- arenas ---------------------------------------------------------------

let weightsBuf: ArrayBuffer = new ArrayBuffer(0);
let weightsBase: usize = 0;

// One arena for every weight tensor. The host copies each tensor to the
// returned bump position (16-byte aligned) and registers the pointers via
// the setLayer/setShared calls.
export function allocWeights(bytes: i32): usize {
  weightsBuf = new ArrayBuffer(bytes);
  weightsBase = changetype<usize>(weightsBuf);
  return weightsBase;
}

// Shared tensors: embedding LN, classifier head.
let embLnW: usize = 0;
let embLnB: usize = 0;
let fc1w: usize = 0;
let fc1b: usize = 0;
let fc2w: usize = 0;
let fc2b: usize = 0;

export function setShared(
  eW: usize, eB: usize, f1w: usize, f1b: usize, f2w: usize, f2b: usize,
): void {
  embLnW = eW;
  embLnB = eB;
  fc1w = f1w;
  fc1b = f1b;
  fc2w = f2w;
  fc2b = f2b;
}

const qkvW = new StaticArray<usize>(LAYERS);
const qkvB = new StaticArray<usize>(LAYERS);
const posK = new StaticArray<usize>(LAYERS);
const posQ = new StaticArray<usize>(LAYERS);
const attW = new StaticArray<usize>(LAYERS);
const attB = new StaticArray<usize>(LAYERS);
const attLnW = new StaticArray<usize>(LAYERS);
const attLnB = new StaticArray<usize>(LAYERS);
const fInW = new StaticArray<usize>(LAYERS);
const fInB = new StaticArray<usize>(LAYERS);
const fOutW = new StaticArray<usize>(LAYERS);
const fOutB = new StaticArray<usize>(LAYERS);
const fLnW = new StaticArray<usize>(LAYERS);
const fLnB = new StaticArray<usize>(LAYERS);

export function setLayer(
  l: i32,
  qW: usize, qB: usize, pK: usize, pQ: usize,
  aW: usize, aB: usize, aW_: usize, aB_: usize,
  iW: usize, iB: usize, oW: usize, oB: usize,
  fW_: usize, fB_: usize,
): void {
  qkvW[l] = qW;
  qkvB[l] = qB;
  posK[l] = pK;
  posQ[l] = pQ;
  attW[l] = aW;
  attB[l] = aB;
  attLnW[l] = aW_;
  attLnB[l] = aB_;
  fInW[l] = iW;
  fInB[l] = iB;
  fOutW[l] = oW;
  fOutB[l] = oB;
  fLnW[l] = fW_;
  fLnB[l] = fB_;
}

// ---- activations and shared input regions ---------------------------------

let _x: ArrayBuffer = new ArrayBuffer(0);
let _tmp: ArrayBuffer = new ArrayBuffer(0);
let _qkv: ArrayBuffer = new ArrayBuffer(0);
let _ctx: ArrayBuffer = new ArrayBuffer(0);
let _attn: ArrayBuffer = new ArrayBuffer(0);
let _mid: ArrayBuffer = new ArrayBuffer(0);
let _ffn: ArrayBuffer = new ArrayBuffer(0);
let _emb: ArrayBuffer = new ArrayBuffer(0);
let _mask: ArrayBuffer = new ArrayBuffer(0);
let _packed: ArrayBuffer = new ArrayBuffer(0);
let _rel: ArrayBuffer = new ArrayBuffer(0);
let _scores: ArrayBuffer = new ArrayBuffer(0);
let _states: ArrayBuffer = new ArrayBuffer(0);
let _h1: ArrayBuffer = new ArrayBuffer(0);
let _raw: ArrayBuffer = new ArrayBuffer(0);
let _logits: ArrayBuffer = new ArrayBuffer(0);

let xP: usize = 0;
let tmpP: usize = 0;
let qkvP: usize = 0;
let ctxP: usize = 0;
let attP: usize = 0;
let midP: usize = 0;
let ffnP: usize = 0;
let embP: usize = 0;
let maskP: usize = 0;
let packedP: usize = 0;
let relP: usize = 0;
let scoresP: usize = 0;
let statesP: usize = 0;
let h1P: usize = 0;
let rawP: usize = 0;
let logitsP: usize = 0;

export function init(bucketLen: i32, eps: f32, temp: f32): void {
  BUCKET = bucketLen;
  EPS = eps;
  TEMP = temp;
  const L = bucketLen;
  _x = new ArrayBuffer(L * HIDDEN * 4);
  _tmp = new ArrayBuffer(L * HIDDEN * 4);
  _qkv = new ArrayBuffer(L * 3 * HIDDEN * 4);
  _ctx = new ArrayBuffer(L * HIDDEN * 4);
  _attn = new ArrayBuffer(L * HIDDEN * 4);
  _mid = new ArrayBuffer(L * INTERMEDIATE * 4);
  _ffn = new ArrayBuffer(L * HIDDEN * 4);
  _emb = new ArrayBuffer(L * HIDDEN * 4);
  _mask = new ArrayBuffer(L * 4);
  _packed = new ArrayBuffer(3 * K_MAX * 4);
  _rel = new ArrayBuffer(L * L * 4);
  _scores = new ArrayBuffer(L * 4);
  _states = new ArrayBuffer(K_MAX * HIDDEN * 4);
  _h1 = new ArrayBuffer(K_MAX * HEAD_HID * 4);
  _raw = new ArrayBuffer(K_MAX * 4);
  _logits = new ArrayBuffer(K_MAX * 4);
  xP = changetype<usize>(_x);
  tmpP = changetype<usize>(_tmp);
  qkvP = changetype<usize>(_qkv);
  ctxP = changetype<usize>(_ctx);
  attP = changetype<usize>(_attn);
  midP = changetype<usize>(_mid);
  ffnP = changetype<usize>(_ffn);
  embP = changetype<usize>(_emb);
  maskP = changetype<usize>(_mask);
  packedP = changetype<usize>(_packed);
  relP = changetype<usize>(_rel);
  scoresP = changetype<usize>(_scores);
  statesP = changetype<usize>(_states);
  h1P = changetype<usize>(_h1);
  rawP = changetype<usize>(_raw);
  logitsP = changetype<usize>(_logits);
}

// Pointers the host writes into before forward() and reads after it.
export function embPtr(): usize { return embP; }
export function maskPtr(): usize { return maskP; }
export function packedPtr(): usize { return packedP; }
export function relidxPtr(): usize { return relP; }
export function logitsPtr(): usize { return logitsP; }
export function xPtr(): usize { return xP; }
export function qkvPtr(): usize { return qkvP; }
export function ctxPtr(): usize { return ctxP; }

// Debug wrappers: expose the primitives for the node bisection probe
// (bench/dbg-wasm.mjs); unused by the shipped forward path.
export function dbgMatmul(
  a: usize, w: usize, bias: usize, c: usize,
  M: i32, N: i32, K: i32, act: i32,
): void {
  matmul(a, w, bias, c, M, N, K, act);
}

export function dbgLnMasked(rows: i32): void {
  lnMasked(rows);
}

export function dbgLnRes(
  a: usize, b: usize, out: usize, w: usize, bias: usize, rows: i32,
): void {
  lnResidualW(a, b, out, w, bias, rows);
}

export function dbgAttention(l: i32, rows: i32, L: i32): void {
  attention(l, rows, L);
}

export function dbgGelu(v: f32): f32 {
  return gelu(v);
}

export function dbgLnStats(a: usize, row: i32, out: usize): void {
  const base = <usize>(row * HIDDEN) * 4;
  var s: f32 = 0;
  var sq: f32 = 0;
  for (let d = 0; d < HIDDEN; d += 1) {
    const v = ldf(a + base, d);
    s += v;
    sq += v * v;
  }
  store<f32>(out, s);
  store<f32>(out + 4, sq);
}

// ---- primitives ------------------------------------------------------------

@inline
function ldf(p: usize, i: i32): f32 {
  return load<f32>(p + <usize>i * 4);
}

@inline
function stf(p: usize, i: i32, v: f32): void {
  store<f32>(p + <usize>i * 4, v);
}

@inline
function hsum(v: v128): f32 {
  return f32x4.extract_lane(v, 0)
    + f32x4.extract_lane(v, 1)
    + f32x4.extract_lane(v, 2)
    + f32x4.extract_lane(v, 3);
}

// C[M,N] = A[M,K] @ W[N,K]^T + B[N]; act: 0 none, 1 relu, 2 gelu (erf via
// Abramowitz-Stegun 7.1.26, identical constants to matmul.wgsl).
// W arrives transposed by the host ([K,N] row-major), so the f32x4 lanes
// run along contiguous output columns and accumulation stays in k order.
@inline
function gelu(v: f32): f32 {
  const u = v * <f32>0.7071067811865476;
  const t = <f32>1.0 / (<f32>1.0 + <f32>0.3275911 * Mathf.abs(u));
  const p = (((((<f32>1.061405429 * t - <f32>1.453152027) * t) + <f32>1.421413741) * t
    - <f32>0.284496736) * t + <f32>0.254829592) * t;
  const e = <f32>1.0 - p * Mathf.exp(-u * u);
  // AS select(true, false, cond) takes the opposite order to WGSL.
  return <f32>(0.5) * v * (<f32>1.0 + select(e, -e, v >= 0.0));
}

function matmul(
  a: usize, w: usize, bias: usize, c: usize,
  M: i32, N: i32, K: i32, act: i32,
): void {
  if (N >= 16) {
    matmulTiled(a, w, bias, c, M, N, K, act);
    return;
  }
  matmulRows(a, w, bias, c, M, N, K, act);
}

// Row-saxpy fallback for narrow outputs (fc2 with N=1 and any N < 16).
function matmulRows(
  a: usize, w: usize, bias: usize, c: usize,
  M: i32, N: i32, K: i32, act: i32,
): void {
  for (let m = 0; m < M; m += 1) {
    const cRow = c + <usize>(m * N) * 4;
    for (let n = 0; n < N; n += 1) {
      store<f32>(cRow + <usize>n * 4, ldf(bias, n));
    }
    const aRow = a + <usize>(m * K) * 4;
    for (let k = 0; k < K; k += 1) {
      const av = v128.splat<f32>(load<f32>(aRow + <usize>k * 4));
      const wRow = w + <usize>k * <usize>N * 4;
      for (let n = 0; n < N; n += 4) {
        const cp = cRow + <usize>n * 4;
        const wv = v128.load(wRow + <usize>n * 4);
        v128.store(cp, f32x4.add(v128.load(cp), f32x4.mul(av, wv)));
      }
    }
    actRow(cRow, N, act);
  }
}

@inline
function actRow(cRow: usize, N: i32, act: i32): void {
  if (act == 1) {
    for (let n = 0; n < N; n += 1) {
      const v = load<f32>(cRow + <usize>n * 4);
      store<f32>(cRow + <usize>n * 4, Mathf.max(v, <f32>0.0));
    }
  } else if (act == 2) {
    for (let n = 0; n < N; n += 1) {
      store<f32>(cRow + <usize>n * 4, gelu(load<f32>(cRow + <usize>n * 4)));
    }
  }
}

@inline
function emitLane(
  cRow: usize, biasP: usize, biasIdx: i32, val: f32, act: i32,
): void {
  let v = val + ldf(biasP, biasIdx);
  if (act == 1) v = Mathf.max(v, <f32>0.0);
  else if (act == 2) v = gelu(v);
  store<f32>(cRow, v);
}

// Lane indices of extract_lane must be immediates, so the four possible
// lanes of a v128 accumulator unroll here; `take` bounds the tail.
@inline
function emitAcc(
  cRow: usize, biasP: usize, biasIdx: i32, acc: v128, take: i32, act: i32,
): void {
  if (take >= 1) {
    emitLane(cRow, biasP, biasIdx, f32x4.extract_lane(acc, 0), act);
    if (take >= 2) {
      emitLane(cRow + 4, biasP, biasIdx + 1, f32x4.extract_lane(acc, 1), act);
      if (take >= 3) {
        emitLane(cRow + 8, biasP, biasIdx + 2, f32x4.extract_lane(acc, 2), act);
        if (take >= 4) {
          emitLane(cRow + 12, biasP, biasIdx + 3, f32x4.extract_lane(acc, 3), act);
        }
      }
    }
  }
}

// 4-row x 16-float micro-kernel: each w block is loaded once for four rows
// and every accumulator stays a named register until the epilogue (a
// StaticArray<v128> would spill them into the k loop). Column tails are
// per-lane guarded; rows beyond M are skipped. W is [K,N] (host-
// transposed); requires N >= 16.
function matmulTiled(
  a: usize, w: usize, bias: usize, c: usize,
  M: i32, N: i32, K: i32, act: i32,
): void {
  for (let mb = 0; mb < M; mb += 4) {
    const rows = min(M - mb, 4);
    for (let nb = 0; nb < N; nb += 16) {
      var a00 = v128.splat<f32>(0); var a01 = a00; var a02 = a00; var a03 = a00;
      var a10 = a00; var a11 = a00; var a12 = a00; var a13 = a00;
      var a20 = a00; var a21 = a00; var a22 = a00; var a23 = a00;
      var a30 = a00; var a31 = a00; var a32 = a00; var a33 = a00;
      for (let k = 0; k < K; k += 1) {
        const wRow = w + <usize>(k * N + nb) * 4;
        const w0 = v128.load(wRow);
        const w1 = v128.load(wRow + 16);
        const w2 = v128.load(wRow + 32);
        const w3 = v128.load(wRow + 48);
        const aRow = a + <usize>(mb * K + k) * 4;
        const v0 = v128.splat<f32>(load<f32>(aRow));
        a00 = f32x4.add(a00, f32x4.mul(v0, w0));
        a01 = f32x4.add(a01, f32x4.mul(v0, w1));
        a02 = f32x4.add(a02, f32x4.mul(v0, w2));
        a03 = f32x4.add(a03, f32x4.mul(v0, w3));
        if (rows >= 2) {
          const v1 = v128.splat<f32>(load<f32>(aRow + <usize>K * 4));
          a10 = f32x4.add(a10, f32x4.mul(v1, w0));
          a11 = f32x4.add(a11, f32x4.mul(v1, w1));
          a12 = f32x4.add(a12, f32x4.mul(v1, w2));
          a13 = f32x4.add(a13, f32x4.mul(v1, w3));
          if (rows >= 3) {
            const v2 = v128.splat<f32>(load<f32>(aRow + <usize>K * 8));
            a20 = f32x4.add(a20, f32x4.mul(v2, w0));
            a21 = f32x4.add(a21, f32x4.mul(v2, w1));
            a22 = f32x4.add(a22, f32x4.mul(v2, w2));
            a23 = f32x4.add(a23, f32x4.mul(v2, w3));
            if (rows >= 4) {
              const v3 = v128.splat<f32>(load<f32>(aRow + <usize>K * 12));
              a30 = f32x4.add(a30, f32x4.mul(v3, w0));
              a31 = f32x4.add(a31, f32x4.mul(v3, w1));
              a32 = f32x4.add(a32, f32x4.mul(v3, w2));
              a33 = f32x4.add(a33, f32x4.mul(v3, w3));
            }
          }
        }
      }
      const lanes = min(N - nb, 16);
      const t0 = min(lanes, 4);
      const t1 = max(min(lanes - 4, 4), 0);
      const t2 = max(min(lanes - 8, 4), 0);
      const t3 = max(min(lanes - 12, 4), 0);
      const bP = bias;
      const bi = nb;
      if (rows >= 1) {
        const r0 = c + <usize>(mb * N + nb) * 4;
        emitAcc(r0, bP, bi, a00, t0, act);
        emitAcc(r0 + 16, bP, bi + 4, a01, t1, act);
        emitAcc(r0 + 32, bP, bi + 8, a02, t2, act);
        emitAcc(r0 + 48, bP, bi + 12, a03, t3, act);
      }
      if (rows >= 2) {
        const r1 = c + <usize>((mb + 1) * N + nb) * 4;
        emitAcc(r1, bP, bi, a10, t0, act);
        emitAcc(r1 + 16, bP, bi + 4, a11, t1, act);
        emitAcc(r1 + 32, bP, bi + 8, a12, t2, act);
        emitAcc(r1 + 48, bP, bi + 12, a13, t3, act);
      }
      if (rows >= 3) {
        const r2 = c + <usize>((mb + 2) * N + nb) * 4;
        emitAcc(r2, bP, bi, a20, t0, act);
        emitAcc(r2 + 16, bP, bi + 4, a21, t1, act);
        emitAcc(r2 + 32, bP, bi + 8, a22, t2, act);
        emitAcc(r2 + 48, bP, bi + 12, a23, t3, act);
      }
      if (rows >= 4) {
        const r3 = c + <usize>((mb + 3) * N + nb) * 4;
        emitAcc(r3, bP, bi, a30, t0, act);
        emitAcc(r3 + 16, bP, bi + 4, a31, t1, act);
        emitAcc(r3 + 32, bP, bi + 8, a32, t2, act);
        emitAcc(r3 + 48, bP, bi + 12, a33, t3, act);
      }
    }
  }
}

// out[i] = (LN(emb[i]) * w + b) * mask[i] — embedding normalization plus mask.
function lnMasked(rows: i32): void {
  for (let i = 0; i < rows; i += 1) {
    const base = <usize>(i * HIDDEN) * 4;
    const mk = ldf(maskP, i);
    var s: f32 = 0;
    var sq: f32 = 0;
    for (let d = 0; d < HIDDEN; d += 1) {
      const v = ldf(embP + base, d);
      s += v;
      sq += v * v;
    }
    const mean = s / <f32>HIDDEN;
    const inv = <f32>1.0 / Mathf.sqrt(sq / <f32>HIDDEN - mean * mean + EPS);
    for (let d = 0; d < HIDDEN; d += 1) {
      const v = ldf(embP + base, d);
      stf(xP + base, d,
        ((v - mean) * inv * ldf(embLnW, d) + ldf(embLnB, d)) * mk);
    }
  }
}

// out[i] = LN(a[i] + b[i]) — residual add fused into LayerNorm.
function lnResidualW(
  a: usize, b: usize, out: usize, w: usize, bias: usize, rows: i32,
): void {
  for (let i = 0; i < rows; i += 1) {
    const base = <usize>(i * HIDDEN) * 4;
    var s: f32 = 0;
    var sq: f32 = 0;
    for (let d = 0; d < HIDDEN; d += 1) {
      const v = ldf(a + base, d) + ldf(b + base, d);
      s += v;
      sq += v * v;
    }
    const mean = s / <f32>HIDDEN;
    const inv = <f32>1.0 / Mathf.sqrt(sq / <f32>HIDDEN - mean * mean + EPS);
    for (let d = 0; d < HIDDEN; d += 1) {
      const v = ldf(a + base, d) + ldf(b + base, d);
      stf(out + base, d, (v - mean) * inv * ldf(w, d) + ldf(bias, d));
    }
  }
}

// Relative attention for one layer: scores = (q.k + c2p + p2c) / SCALE,
// masked softmax, weighted sum over v. Mirrors attention.wgsl. L is the
// request bucket; relidx is stored at BUCKET stride (one 1024 table covers
// every bucket, values depend only on the i-j delta).
function attention(l: i32, rows: i32, L: i32): void {
  const SCALE: f32 = 13.856406460551018; // f32 sqrt(3 * 64), same as attention.wgsl
  for (let h = 0; h < HEADS; h += 1) {
    for (let i = 0; i < rows; i += 1) {
      const qBase = qkvP + <usize>(i * 3 * HD + h * HEAD_DIM) * 4;
      if (ldf(maskP, i) <= 0.5) {
        for (let d = 0; d < HEAD_DIM; d += 1) {
          stf(ctxP + <usize>(i * HD + h * HEAD_DIM) * 4, d, 0.0);
        }
        continue;
      }
      for (let j = 0; j < L; j += 1) {
        const p = <usize>load<u32>(relP + <usize>(i * BUCKET + j) * 4);
        const kBase = qkvP + <usize>(j * 3 * HD + (HEADS + h) * HEAD_DIM) * 4;
        const pkBase = posK[l] + (p * <usize>HD + <usize>(h * HEAD_DIM)) * 4;
        const pqBase = posQ[l] + (p * <usize>HD + <usize>(h * HEAD_DIM)) * 4;
        var a1 = v128.splat<f32>(0);
        var a2 = v128.splat<f32>(0);
        var a3 = v128.splat<f32>(0);
        for (let d = 0; d < HEAD_DIM; d += 4) {
          const qv = v128.load(qBase + <usize>d * 4);
          const kv = v128.load(kBase + <usize>d * 4);
          const pv = v128.load(pkBase + <usize>d * 4);
          const rv = v128.load(pqBase + <usize>d * 4);
          a1 = f32x4.add(a1, f32x4.mul(qv, kv));
          a2 = f32x4.add(a2, f32x4.mul(qv, pv));
          a3 = f32x4.add(a3, f32x4.mul(kv, rv));
        }
        const s = (hsum(a1) + hsum(a2) + hsum(a3)) / SCALE;
        stf(scoresP, j, ldf(maskP, j) > 0.5 ? s : <f32>-1e4);
      }
      var mx: f32 = <f32>-1e30;
      for (let j = 0; j < L; j += 1) mx = Mathf.max(mx, ldf(scoresP, j));
      var sum: f32 = 0;
      for (let j = 0; j < L; j += 1) {
        const e = Mathf.exp(ldf(scoresP, j) - mx);
        stf(scoresP, j, e);
        sum += e;
      }
      for (let j = 0; j < L; j += 1) {
        stf(scoresP, j, ldf(scoresP, j) / sum);
      }
      // ctx[i, h*D + d] = sum_j scores[j] * v[j, (2H+h)*D + d]
      const ctxRow = ctxP + <usize>(i * HD + h * HEAD_DIM) * 4;
      for (let d = 0; d < HEAD_DIM; d += 4) {
        v128.store(ctxRow + <usize>d * 4, v128.splat<f32>(0));
      }
      for (let j = 0; j < L; j += 1) {
        const sv = ldf(scoresP, j);
        if (sv == 0.0) continue;
        const vBase = qkvP + <usize>(j * 3 * HD + (2 * HEADS + h) * HEAD_DIM) * 4;
        const svv = v128.splat<f32>(sv);
        for (let d = 0; d < HEAD_DIM; d += 4) {
          const cp = ctxRow + <usize>d * 4;
          v128.store(cp, f32x4.add(
            v128.load(cp), f32x4.mul(svv, v128.load(vBase + <usize>d * 4))));
        }
      }
    }
  }
}

// ---- forward ---------------------------------------------------------------

export function forward(seqLen: i32, L: i32): void {
  lnMasked(seqLen);
  for (let l = 0; l < LAYERS; l += 1) {
    matmul(xP, qkvW[l], qkvB[l], qkvP, seqLen, 3 * HIDDEN, HIDDEN, 0);
    attention(l, seqLen, L);
    matmul(ctxP, attW[l], attB[l], attP, seqLen, HIDDEN, HIDDEN, 0);
    lnResidualW(xP, attP, tmpP, attLnW[l], attLnB[l], seqLen);
    matmul(tmpP, fInW[l], fInB[l], midP, seqLen, INTERMEDIATE, HIDDEN, 2);
    matmul(midP, fOutW[l], fOutB[l], ffnP, seqLen, HIDDEN, INTERMEDIATE, 0);
    lnResidualW(tmpP, ffnP, xP, fLnW[l], fLnB[l], seqLen);
  }
  // Head: gather marker states, fc1 + relu, fc2, marker mask on logits.
  for (let k = 0; k < K_MAX; k += 1) {
    const mk = load<u32>(packedP + <usize>k * 4);
    const src = xP + <usize>(<i32>mk * HIDDEN) * 4;
    const dst = statesP + <usize>(k * HIDDEN) * 4;
    memory.copy(dst, src, <usize>HIDDEN * 4);
  }
  matmul(statesP, fc1w, fc1b, h1P, K_MAX, HEAD_HID, HIDDEN, 1);
  matmul(h1P, fc2w, fc2b, rawP, K_MAX, 1, HEAD_HID, 0);
  for (let k = 0; k < K_MAX; k += 1) {
    const m = load<f32>(packedP + <usize>(K_MAX + k) * 4);
    stf(logitsP, k, m > 0.5 ? ldf(rawP, k) / TEMP : <f32>-1e4);
  }
}
