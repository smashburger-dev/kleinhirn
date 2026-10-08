// WASM executor for a Plan (R2, docs/R2_WORKORDER.md Festlegung 1): one function per kind of
// computation, not per kernel name. The host (src/plan/wasm.ts) translates the op list of a
// plan once into a command stream in this module's memory and places weights and plan buffers
// at fixed addresses; run(stream, rows) executes the whole stream, one call per forward.
// f32 storage and f32 accumulation throughout (Festlegung 3). The results follow the WGSL
// kernels in src/kernels/ (same formulas, same masking), not their bits: sums run in another
// order. Scratch addresses come from the host; the module allocates nothing.
//
// Stream: header of HEAD words, then one record of OPW words per op:
//   header [ops, L, batch, mask, kinfo, scratch, scratch bytes per thread, ctl (0: one thread)]
//   record [code, rowsFixed, arg0, arg1, ...]; rowsFixed >= 0 is the row count of an op with a
//   fixed dispatch (pooled heads, markers), -1 takes the rows of the call.
// Prefix masks: a sequence's valid keys are first .. last without holes (EncoderModel, GLiNER,
// Julia fill masks that way), so the attention reads keys first .. last only.

const HEAD: i32 = 8;
const OPW: i32 = 32;

const GEMM_P: i32 = 1;     // C = act(A B + bias), B in panels [N / 16][K][16] (packed at load)
const GEMM_NT: i32 = 2;    // C = act(alpha A B^T + bias), B [N, K]
const LN: i32 = 3;
const EMBLN: i32 = 4;
const ADD: i32 = 5;
const GATHER: i32 = 6;
const POOL: i32 = 7;
const GEGLU: i32 = 8;
const ROPE: i32 = 9;
const IM2COL: i32 = 10;
const ATTSCORE: i32 = 11;
const ATTSOFTMAX: i32 = 12;
const ATTPV: i32 = 13;
const ATTREL: i32 = 14;
const ATTSOFTREL: i32 = 15;
const ATTSTD: i32 = 16;    // mbattention, mbflash
const ATTRELF: i32 = 17;   // attention (DeBERTa, fused)
const MASKLOGITS: i32 = 18;

// lowest finite f32: start of a softmax maximum (review R21)
const LOWEST: f32 = -3.40282346638528859812e38;

export function heapBase(): usize {
  return __heap_base;
}

// ---- helpers ---------------------------------------------------------------

@inline function ldf(p: usize, i: i32): f32 { return load<f32>(p + (<usize>i << 2)); }
@inline function stf(p: usize, i: i32, v: f32): void { store<f32>(p + (<usize>i << 2), v); }
@inline function ldi(p: usize, i: i32): i32 { return load<i32>(p + (<usize>i << 2)); }
@inline function off(p: usize, i: i32): usize { return p + (<usize>i << 2); }

@inline function hsum(v: v128): f32 {
  return f32x4.extract_lane(v, 0) + f32x4.extract_lane(v, 1)
    + f32x4.extract_lane(v, 2) + f32x4.extract_lane(v, 3);
}

// e^x on four lanes (R8 hc10): x = n ln2 + r with |r| <= ln2 / 2, e^r by the minimax polynomial
// of Cephes expf, 2^n through the exponent bits. Inputs are clamped to [-87.3, 88.3]: below, the
// result is about 1e-38 instead of 0. One code for vector and tail lanes (vexp1), so a value does
// not depend on its position in a row.
@inline function vexp(x: v128): v128 {
  x = f32x4.max(f32x4.min(x, f32x4.splat(<f32>88.3762626647949)), f32x4.splat(<f32>-87.3365447504));
  const n = f32x4.floor(f32x4.add(f32x4.mul(x, f32x4.splat(<f32>1.44269504088896341)), f32x4.splat(<f32>0.5)));
  let r = f32x4.sub(x, f32x4.mul(n, f32x4.splat(<f32>0.693359375)));
  r = f32x4.sub(r, f32x4.mul(n, f32x4.splat(<f32>-2.12194440e-4)));
  let y = f32x4.splat(<f32>1.9875691500e-4);
  y = f32x4.add(f32x4.mul(y, r), f32x4.splat(<f32>1.3981999507e-3));
  y = f32x4.add(f32x4.mul(y, r), f32x4.splat(<f32>8.3334519073e-3));
  y = f32x4.add(f32x4.mul(y, r), f32x4.splat(<f32>4.1665795894e-2));
  y = f32x4.add(f32x4.mul(y, r), f32x4.splat(<f32>1.6666665459e-1));
  y = f32x4.add(f32x4.mul(y, r), f32x4.splat(<f32>5.0000001201e-1));
  y = f32x4.add(f32x4.add(f32x4.mul(f32x4.mul(y, r), r), r), f32x4.splat(<f32>1.0));
  const pow2n = i32x4.shl(i32x4.add(i32x4.trunc_sat_f32x4_s(n), i32x4.splat(127)), 23);
  return f32x4.mul(y, pow2n);
}

@inline function vexp1(x: f32): f32 {
  return f32x4.extract_lane(vexp(f32x4.splat(x)), 0);
}

// GELU on four lanes with vexp (R8 hc10), the erf approximation of gelu().
@inline function vgelu(v: v128): v128 {
  const u = f32x4.mul(v, f32x4.splat(<f32>0.7071067811865476));
  const t = f32x4.div(f32x4.splat(<f32>1.0), f32x4.add(f32x4.splat(<f32>1.0), f32x4.mul(f32x4.splat(<f32>0.3275911), f32x4.abs(u))));
  let p = f32x4.sub(f32x4.mul(f32x4.splat(<f32>1.061405429), t), f32x4.splat(<f32>1.453152027));
  p = f32x4.add(f32x4.mul(p, t), f32x4.splat(<f32>1.421413741));
  p = f32x4.sub(f32x4.mul(p, t), f32x4.splat(<f32>0.284496736));
  p = f32x4.add(f32x4.mul(p, t), f32x4.splat(<f32>0.254829592));
  p = f32x4.mul(p, t);
  const e = f32x4.sub(f32x4.splat(<f32>1.0), f32x4.mul(p, vexp(f32x4.neg(f32x4.mul(u, u)))));
  const se = v128.bitselect(e, f32x4.neg(e), f32x4.ge(v, f32x4.splat(0)));
  return f32x4.mul(f32x4.mul(f32x4.splat(<f32>0.5), v), f32x4.add(f32x4.splat(<f32>1.0), se));
}

@inline function vgelu1(v: f32): f32 {
  return f32x4.extract_lane(vgelu(f32x4.splat(v)), 0);
}

// GELU via erf, Abramowitz-Stegun 7.1.26, the constants of matmul.wgsl.
@inline function gelu(v: f32): f32 {
  const u = v * <f32>0.7071067811865476;
  const t = <f32>1.0 / (<f32>1.0 + <f32>0.3275911 * Mathf.abs(u));
  const p = (((((<f32>1.061405429 * t - <f32>1.453152027) * t) + <f32>1.421413741) * t
    - <f32>0.284496736) * t + <f32>0.254829592) * t;
  const e = <f32>1.0 - p * Mathf.exp(-u * u);
  return <f32>0.5 * v * (<f32>1.0 + select<f32>(e, -e, v >= 0));
}

// ACT: 0 none, 1 relu, 2 gelu, 3 tanh, 4 silu (matmul.wgsl).
function activate(v: f32, act: i32): f32 {
  if (act == 1) return Mathf.max(v, 0);
  if (act == 2) return gelu(v);
  if (act == 3) return Mathf.tanh(v);
  if (act == 4) return v / (<f32>1.0 + Mathf.exp(-v));
  return v;
}

@inline function fin(acc: f32, alpha: f32, bias: usize, n: i32, act: i32): f32 {
  let v = acc * alpha;
  if (bias) v += ldf(bias, n);
  return act ? activate(v, act) : v;
}

// ---- matrix products ---------------------------------------------------------

// Bias and activation over an M x N block of C (row stride ldc), after a matmul that stored the
// plain sums: the arithmetic of fin() with alpha 1, so the bits do not change. A separate pass
// keeps the epilogue out of the matmul function (R8 hc1: JavaScriptCore and SpiderMonkey ran the
// k loop at half speed while the function also held 16 inlined epilogues, FINDINGS §55).
function biasAct(c: usize, ldc: i32, bias: usize, M: i32, N: i32, act: i32): void {
  for (let m = 0; m < M; m += 1) {
    const r = off(c, m * ldc);
    let n0 = 0;
    if (act == 2 || act == 0) {
      // bias and GELU on four lanes (R8 hc10); the tail with the same formula
      for (; n0 + 4 <= N; n0 += 4) {
        let v = v128.load(off(r, n0));
        if (bias) v = f32x4.add(v, v128.load(off(bias, n0)));
        v128.store(off(r, n0), act ? vgelu(v) : v);
      }
      for (; n0 < N; n0 += 1) {
        let v = ldf(r, n0);
        if (bias) v += ldf(bias, n0);
        stf(r, n0, act ? vgelu1(v) : v);
      }
      continue;
    }
    for (let n = 0; n < N; n += 1) {
      let v = ldf(r, n);
      if (bias) v += ldf(bias, n);
      stf(r, n, act ? activate(v, act) : v);
    }
  }
}

@inline function storeAcc(c: usize, acc: v128, take: i32): void {
  if (take >= 4) { v128.store(c, acc); return; }
  if (take >= 1) stf(c, 0, f32x4.extract_lane(acc, 0));
  if (take >= 2) stf(c, 1, f32x4.extract_lane(acc, 1));
  if (take >= 3) stf(c, 2, f32x4.extract_lane(acc, 2));
}

@inline function storeRow(c: usize, lanes: i32, a0: v128, a1: v128, a2: v128, a3: v128): void {
  storeAcc(c, a0, lanes);
  storeAcc(c + 16, a1, lanes - 4);
  storeAcc(c + 32, a2, lanes - 8);
  storeAcc(c + 48, a3, lanes - 12);
}

// C[m, n] = sum_k A[m, k] B[k, n], B row-major [K, N] with stride ldb; bias and activation are
// biasAct's. 4 rows x 16 columns per tile (the kernel of the former src/wasm/deberta.ts, with
// strides); the 16 accumulators stay in registers over k. A column tail loads the full 16 lanes
// and stores only the valid ones: every buffer ends in 64 bytes of padding.
function gemmNN(a: usize, lda: i32, b: usize, ldb: i32, c: usize, ldc: i32,
  M: i32, N: i32, K: i32): void {
  const sa = <usize>lda << 2;
  for (let mb = 0; mb < M; mb += 4) {
    const rows = min(M - mb, 4);
    const aRow = a + (<usize>(mb * lda) << 2);
    for (let nb = 0; nb < N; nb += 16) {
      let a00 = f32x4.splat(0); let a01 = a00; let a02 = a00; let a03 = a00;
      let a10 = a00; let a11 = a00; let a12 = a00; let a13 = a00;
      let a20 = a00; let a21 = a00; let a22 = a00; let a23 = a00;
      let a30 = a00; let a31 = a00; let a32 = a00; let a33 = a00;
      let bp = b + (<usize>nb << 2);
      const sb = <usize>ldb << 2;
      for (let k = 0; k < K; k += 1) {
        const w0 = v128.load(bp);
        const w1 = v128.load(bp, 16);
        const w2 = v128.load(bp, 32);
        const w3 = v128.load(bp, 48);
        bp += sb;
        const ap = aRow + (<usize>k << 2);
        const v0 = f32x4.splat(load<f32>(ap));
        a00 = madd(v0, w0, a00);
        a01 = madd(v0, w1, a01);
        a02 = madd(v0, w2, a02);
        a03 = madd(v0, w3, a03);
        if (rows >= 2) {
          const v1 = f32x4.splat(load<f32>(ap + sa));
          a10 = madd(v1, w0, a10);
          a11 = madd(v1, w1, a11);
          a12 = madd(v1, w2, a12);
          a13 = madd(v1, w3, a13);
          if (rows >= 3) {
            const v2 = f32x4.splat(load<f32>(ap + 2 * sa));
            a20 = madd(v2, w0, a20);
            a21 = madd(v2, w1, a21);
            a22 = madd(v2, w2, a22);
            a23 = madd(v2, w3, a23);
            if (rows >= 4) {
              const v3 = f32x4.splat(load<f32>(ap + 3 * sa));
              a30 = madd(v3, w0, a30);
              a31 = madd(v3, w1, a31);
              a32 = madd(v3, w2, a32);
              a33 = madd(v3, w3, a33);
            }
          }
        }
      }
      const lanes = min(N - nb, 16);
      const c0 = c + (<usize>(mb * ldc + nb) << 2);
      const sc = <usize>ldc << 2;
      storeRow(c0, lanes, a00, a01, a02, a03);
      if (rows >= 2) storeRow(c0 + sc, lanes, a10, a11, a12, a13);
      if (rows >= 3) storeRow(c0 + 2 * sc, lanes, a20, a21, a22, a23);
      if (rows >= 4) storeRow(c0 + 3 * sc, lanes, a30, a31, a32, a33);
    }
  }
}

// Multiply-add of the weight matmul: mul, then add of the accumulator (IEEE add commutes, so the
// plain build keeps the bits of add(acc, mul)). The relaxed build (plan-relaxed.wasm, R8 hc6)
// uses f32x4.relaxed_madd, one rounding where the engine fuses it.
@inline function madd(x: v128, w: v128, acc: v128): v128 {
  if (ASC_FEATURE_RELAXED_SIMD) return f32x4.relaxed_madd(x, w, acc);
  return f32x4.add(f32x4.mul(x, w), acc);
}

// C[m, n] = sum_k A[m, k] B[k, n] for a weight B packed in panels [ceil(N / 16)][K][16] (R8 hc2):
// each k reads one contiguous 64-byte panel row. Full 4-row tiles without row checks in the k
// loop, then single rows; columns past N are zero in the panel and are not stored. Bias and
// activation are biasAct's.
function gemmP(a: usize, lda: i32, p: usize, c: usize, ldc: i32, M: i32, N: i32, K: i32): void {
  const sa = <usize>lda << 2;
  const sc = <usize>ldc << 2;
  const ps = <usize>K << 6; // bytes per panel
  let m = 0;
  for (; m + 4 <= M; m += 4) {
    const ar = a + <usize>m * sa;
    for (let nb = 0; nb < N; nb += 16) {
      let a00 = f32x4.splat(0); let a01 = a00; let a02 = a00; let a03 = a00;
      let a10 = a00; let a11 = a00; let a12 = a00; let a13 = a00;
      let a20 = a00; let a21 = a00; let a22 = a00; let a23 = a00;
      let a30 = a00; let a31 = a00; let a32 = a00; let a33 = a00;
      let bp = p + <usize>(nb >> 4) * ps;
      for (let k = 0; k < K; k += 1) {
        const w0 = v128.load(bp);
        const w1 = v128.load(bp, 16);
        const w2 = v128.load(bp, 32);
        const w3 = v128.load(bp, 48);
        bp += 64;
        const q = ar + (<usize>k << 2);
        const x0 = f32x4.splat(load<f32>(q));
        a00 = madd(x0, w0, a00); a01 = madd(x0, w1, a01); a02 = madd(x0, w2, a02); a03 = madd(x0, w3, a03);
        const x1 = f32x4.splat(load<f32>(q + sa));
        a10 = madd(x1, w0, a10); a11 = madd(x1, w1, a11); a12 = madd(x1, w2, a12); a13 = madd(x1, w3, a13);
        const x2 = f32x4.splat(load<f32>(q + 2 * sa));
        a20 = madd(x2, w0, a20); a21 = madd(x2, w1, a21); a22 = madd(x2, w2, a22); a23 = madd(x2, w3, a23);
        const x3 = f32x4.splat(load<f32>(q + 3 * sa));
        a30 = madd(x3, w0, a30); a31 = madd(x3, w1, a31); a32 = madd(x3, w2, a32); a33 = madd(x3, w3, a33);
      }
      const lanes = min(N - nb, 16);
      const c0 = c + <usize>m * sc + (<usize>nb << 2);
      storeRow(c0, lanes, a00, a01, a02, a03);
      storeRow(c0 + sc, lanes, a10, a11, a12, a13);
      storeRow(c0 + 2 * sc, lanes, a20, a21, a22, a23);
      storeRow(c0 + 3 * sc, lanes, a30, a31, a32, a33);
    }
  }
  for (; m < M; m += 1) {
    const ar = a + <usize>m * sa;
    for (let nb = 0; nb < N; nb += 16) {
      let s0 = f32x4.splat(0); let s1 = s0; let s2 = s0; let s3 = s0;
      let bp = p + <usize>(nb >> 4) * ps;
      for (let k = 0; k < K; k += 1) {
        const x = f32x4.splat(load<f32>(ar + (<usize>k << 2)));
        s0 = madd(x, v128.load(bp), s0); s1 = madd(x, v128.load(bp, 16), s1);
        s2 = madd(x, v128.load(bp, 32), s2); s3 = madd(x, v128.load(bp, 48), s3);
        bp += 64;
      }
      storeRow(c + <usize>m * sc + (<usize>nb << 2), min(N - nb, 16), s0, s1, s2, s3);
    }
  }
}

// sum_k a[k] w[k] over K values, four lanes and a scalar tail.
function dot(a: usize, w: usize, K: i32): f32 {
  let acc = f32x4.splat(0);
  let k = 0;
  for (; k + 4 <= K; k += 4) {
    acc = f32x4.add(acc, f32x4.mul(v128.load(off(a, k)), v128.load(off(w, k))));
  }
  let s = hsum(acc);
  for (; k < K; k += 1) s += ldf(a, k) * ldf(w, k);
  return s;
}

// C[m, n] = act(alpha * sum_k A[m, k] B[n, k] + bias[n]), B row-major [N, K] with stride ldb:
// dot products along k, 4 x 4 outputs per tile, edges one dot product each.
function gemmNT(a: usize, lda: i32, b: usize, ldb: i32, c: usize, ldc: i32, bias: usize,
  M: i32, N: i32, K: i32, alpha: f32, act: i32): void {
  const K4 = K & ~3;
  let m = 0;
  for (; m + 4 <= M; m += 4) {
    const r0 = a + (<usize>(m * lda) << 2);
    const r1 = r0 + (<usize>lda << 2);
    const r2 = r1 + (<usize>lda << 2);
    const r3 = r2 + (<usize>lda << 2);
    let n = 0;
    for (; n + 4 <= N; n += 4) {
      const q0 = b + (<usize>(n * ldb) << 2);
      const q1 = q0 + (<usize>ldb << 2);
      const q2 = q1 + (<usize>ldb << 2);
      const q3 = q2 + (<usize>ldb << 2);
      let c00 = f32x4.splat(0); let c01 = c00; let c02 = c00; let c03 = c00;
      let c10 = c00; let c11 = c00; let c12 = c00; let c13 = c00;
      let c20 = c00; let c21 = c00; let c22 = c00; let c23 = c00;
      let c30 = c00; let c31 = c00; let c32 = c00; let c33 = c00;
      for (let k = 0; k < K4; k += 4) {
        const o = <usize>k << 2;
        const x0 = v128.load(r0 + o); const x1 = v128.load(r1 + o);
        const x2 = v128.load(r2 + o); const x3 = v128.load(r3 + o);
        const y0 = v128.load(q0 + o); const y1 = v128.load(q1 + o);
        const y2 = v128.load(q2 + o); const y3 = v128.load(q3 + o);
        c00 = f32x4.add(c00, f32x4.mul(x0, y0)); c01 = f32x4.add(c01, f32x4.mul(x0, y1));
        c02 = f32x4.add(c02, f32x4.mul(x0, y2)); c03 = f32x4.add(c03, f32x4.mul(x0, y3));
        c10 = f32x4.add(c10, f32x4.mul(x1, y0)); c11 = f32x4.add(c11, f32x4.mul(x1, y1));
        c12 = f32x4.add(c12, f32x4.mul(x1, y2)); c13 = f32x4.add(c13, f32x4.mul(x1, y3));
        c20 = f32x4.add(c20, f32x4.mul(x2, y0)); c21 = f32x4.add(c21, f32x4.mul(x2, y1));
        c22 = f32x4.add(c22, f32x4.mul(x2, y2)); c23 = f32x4.add(c23, f32x4.mul(x2, y3));
        c30 = f32x4.add(c30, f32x4.mul(x3, y0)); c31 = f32x4.add(c31, f32x4.mul(x3, y1));
        c32 = f32x4.add(c32, f32x4.mul(x3, y2)); c33 = f32x4.add(c33, f32x4.mul(x3, y3));
      }
      const o0 = c + (<usize>(m * ldc + n) << 2);
      const sc = <usize>ldc << 2;
      ntStore(o0, r0, q0, q1, q2, q3, K4, K, c00, c01, c02, c03, alpha, bias, n, act);
      ntStore(o0 + sc, r1, q0, q1, q2, q3, K4, K, c10, c11, c12, c13, alpha, bias, n, act);
      ntStore(o0 + 2 * sc, r2, q0, q1, q2, q3, K4, K, c20, c21, c22, c23, alpha, bias, n, act);
      ntStore(o0 + 3 * sc, r3, q0, q1, q2, q3, K4, K, c30, c31, c32, c33, alpha, bias, n, act);
    }
    for (; n < N; n += 1) {
      const q = b + (<usize>(n * ldb) << 2);
      const o0 = c + (<usize>(m * ldc + n) << 2);
      const sc = <usize>ldc << 2;
      store<f32>(o0, fin(dot(r0, q, K), alpha, bias, n, act));
      store<f32>(o0 + sc, fin(dot(r1, q, K), alpha, bias, n, act));
      store<f32>(o0 + 2 * sc, fin(dot(r2, q, K), alpha, bias, n, act));
      store<f32>(o0 + 3 * sc, fin(dot(r3, q, K), alpha, bias, n, act));
    }
  }
  for (; m < M; m += 1) {
    const r = a + (<usize>(m * lda) << 2);
    for (let n = 0; n < N; n += 1) {
      stf(c, m * ldc + n, fin(dot(r, b + (<usize>(n * ldb) << 2), K), alpha, bias, n, act));
    }
  }
}

@inline function tail(x: usize, y: usize, k0: i32, K: i32): f32 {
  let s: f32 = 0;
  for (let k = k0; k < K; k += 1) s += ldf(x, k) * ldf(y, k);
  return s;
}

@inline function ntStore(o: usize, x: usize, q0: usize, q1: usize, q2: usize, q3: usize, K4: i32, K: i32,
  v0: v128, v1: v128, v2: v128, v3: v128, alpha: f32, bias: usize, n: i32, act: i32): void {
  stf(o, 0, fin(hsum(v0) + tail(x, q0, K4, K), alpha, bias, n, act));
  stf(o, 1, fin(hsum(v1) + tail(x, q1, K4, K), alpha, bias, n + 1, act));
  stf(o, 2, fin(hsum(v2) + tail(x, q2, K4, K), alpha, bias, n + 2, act));
  stf(o, 3, fin(hsum(v3) + tail(x, q3, K4, K), alpha, bias, n + 3, act));
}

// ---- normalization ----------------------------------------------------------

// Row o holds v; afterwards ((v - mean) * inv * w + bias) * scale with the variance as the mean
// of (v - mean)^2 (two passes, review R05, as layernorm.wgsl).
function normRow(o: usize, N: i32, w: usize, bias: usize, eps: f32, scale: f32): void {
  let acc = f32x4.splat(0);
  let d = 0;
  for (; d + 4 <= N; d += 4) acc = f32x4.add(acc, v128.load(off(o, d)));
  let s = hsum(acc);
  for (; d < N; d += 1) s += ldf(o, d);
  const mean = s / <f32>N;
  const mv = f32x4.splat(mean);
  acc = f32x4.splat(0);
  d = 0;
  for (; d + 4 <= N; d += 4) {
    const x = f32x4.sub(v128.load(off(o, d)), mv);
    acc = f32x4.add(acc, f32x4.mul(x, x));
  }
  let sq = hsum(acc);
  for (; d < N; d += 1) {
    const x = ldf(o, d) - mean;
    sq += x * x;
  }
  const inv = <f32>1.0 / Mathf.sqrt(sq / <f32>N + eps);
  for (d = 0; d < N; d += 1) {
    stf(o, d, ((ldf(o, d) - mean) * inv * ldf(w, d) + ldf(bias, d)) * scale);
  }
}

// MODE 0 LN(a), 1 LN(a) * mask[row], 2 LN(a + b) (layernorm.wgsl), rows r0 .. r1-1.
function layernorm(a: usize, b: usize, w: usize, bias: usize, mask: usize, out: usize,
  N: i32, mode: i32, eps: f32, r0: i32, r1: i32): void {
  for (let r = r0; r < r1; r += 1) {
    const o = off(out, r * N);
    const x = off(a, r * N);
    if (mode == 2) {
      const y = off(b, r * N);
      let d = 0;
      for (; d + 4 <= N; d += 4) {
        v128.store(off(o, d), f32x4.add(v128.load(off(x, d)), v128.load(off(y, d))));
      }
      for (; d < N; d += 1) stf(o, d, ldf(x, d) + ldf(y, d));
    } else {
      memory.copy(o, x, <usize>N << 2);
    }
    normRow(o, N, w, bias, eps, mode == 1 ? ldf(mask, r) : <f32>1.0);
  }
}

// word + position + type, LayerNorm, mask (embln.wgsl).
function embln(word: usize, pos: usize, typ: usize, tt: usize, w: usize, bias: usize, mask: usize,
  out: usize, N: i32, L: i32, offset: i32, maxpos: i32, eps: f32, maskmul: i32, posids: i32,
  rows: i32): void {
  for (let r = 0; r < rows; r += 1) {
    const t = <u32>ldi(tt, r);
    const prow = posids ? <i32>(t >> 16) : (r % L) + offset;
    const p = off(pos, min(prow, maxpos - 1) * N);
    const y = off(typ, <i32>(t & 0xffff) * N);
    const x = off(word, r * N);
    const o = off(out, r * N);
    for (let d = 0; d < N; d += 1) stf(o, d, ldf(x, d) + ldf(p, d) + ldf(y, d));
    normRow(o, N, w, bias, eps, maskmul ? ldf(mask, r) : <f32>1.0);
  }
}

// ---- small elementwise ops ---------------------------------------------------

// MODE 0 dst += src, MODE 1 dst[i] += src[(i / (N L)) N + i % N] (add.wgsl).
function add(dst: usize, src: usize, count: i32, N: i32, mode: i32, L: i32): void {
  if (mode == 0) {
    let i = 0;
    for (; i + 4 <= count; i += 4) {
      v128.store(off(dst, i), f32x4.add(v128.load(off(dst, i)), v128.load(off(src, i))));
    }
    for (; i < count; i += 1) stf(dst, i, ldf(dst, i) + ldf(src, i));
    return;
  }
  for (let i = 0; i < count; i += 1) {
    stf(dst, i, ldf(dst, i) + ldf(src, (i / (N * L)) * N + i % N));
  }
}

function gather(markers: usize, x: usize, states: usize, K: i32, D: i32, L: i32, total: i32): void {
  for (let g = 0; g < total; g += 1) {
    const b = g / K;
    const row = b * L + ldi(markers, b * 3 * K + (g - b * K));
    memory.copy(off(states, g * D), off(x, row * D), <usize>D << 2);
  }
}

// MODE 0 masked mean (floor 1e-9), MODE 1 masked max (pool.wgsl); per column the rows in order.
function pool(x: usize, mask: usize, out: usize, L: i32, N: i32, mode: i32, batch: i32): void {
  for (let b = 0; b < batch; b += 1) {
    const o = off(out, b * N);
    for (let d = 0; d < N; d += 1) stf(o, d, 0);
    let cnt: f32 = 0;
    let found = false;
    for (let i = 0; i < L; i += 1) {
      const m = ldf(mask, b * L + i);
      const xr = off(x, (b * L + i) * N);
      if (mode == 0) {
        if (m == 0) continue;
        for (let d = 0; d < N; d += 1) stf(o, d, ldf(o, d) + m * ldf(xr, d));
        cnt += m;
      } else if (m > 0.5) {
        for (let d = 0; d < N; d += 1) stf(o, d, found ? Mathf.max(ldf(o, d), ldf(xr, d)) : ldf(xr, d));
        found = true;
      }
    }
    if (mode == 0) {
      const den = Mathf.max(cnt, <f32>1e-9);
      for (let d = 0; d < N; d += 1) stf(o, d, ldf(o, d) / den);
    }
  }
}

function geglu(mid: usize, gate: usize, I: i32, r0: i32, r1: i32): void {
  for (let r = r0; r < r1; r += 1) {
    const m = off(mid, r * 2 * I);
    const g = off(gate, r * I);
    let j = 0;
    for (; j + 4 <= I; j += 4) v128.store(off(g, j), f32x4.mul(vgelu(v128.load(off(m, j))), v128.load(off(m, I + j))));
    for (; j < I; j += 1) stf(g, j, vgelu1(ldf(m, j)) * ldf(m, I + j));
  }
}

// rotate-half RoPE on the q and k parts of qkv, table [L][cos D | sin D] (rope.wgsl).
function rope(qkv: usize, cossin: usize, L: i32, H: i32, D: i32, rows: i32): void {
  const hd = H * D;
  const half = D / 2;
  for (let i = 0; i < rows; i += 1) {
    const t = off(cossin, (i % L) * 2 * D);
    for (let s = 0; s < 2 * H; s += 1) {
      const base = off(qkv, i * 3 * hd + s * D);
      for (let d = 0; d < half; d += 1) {
        const x = ldf(base, d);
        const y = ldf(base, d + half);
        const c = ldf(t, d);
        const sn = ldf(t, D + d);
        stf(base, d, x * c - y * sn);
        stf(base, d + half, y * c + x * sn);
      }
    }
  }
}

function im2col(emb: usize, mask: usize, out: usize, N: i32, L: i32, KS: i32, rows: i32): void {
  const pad = (KS - 1) / 2;
  for (let r = 0; r < rows; r += 1) {
    const seq = r / L;
    const i = r % L;
    for (let t = 0; t < KS; t += 1) {
      const j = i + t - pad;
      const src = seq * L + max(j, 0);
      const o = off(out, r * KS * N + t * N);
      if (j >= 0 && j < L && ldf(mask, src) != 0) memory.copy(o, off(emb, src * N), <usize>N << 2);
      else memory.fill(o, 0, <usize>N << 2);
    }
  }
}

function masklogits(raw: usize, packed: usize, logits: usize, K: i32, temp: f32, total: i32): void {
  for (let k = 0; k < total; k += 1) {
    const b = k / K;
    const m = ldf(packed, b * 3 * K + K + (k - b * K));
    stf(logits, k, m > 0.5 ? ldf(raw, k) / temp : <f32>-1e4);
  }
}

// ---- attention ---------------------------------------------------------------

// first and last valid key per sequence, [first, last] pairs at kinfo; first = L when none
function keyRanges(mask: usize, kinfo: usize, L: i32, batch: i32): void {
  for (let b = 0; b < batch; b += 1) {
    let first = L;
    let last = 0;
    for (let j = 0; j < L; j += 1) {
      if (ldf(mask, b * L + j) > 0.5) {
        if (first == L) first = j;
        last = j;
      }
    }
    store<i32>(kinfo + (<usize>(2 * b) << 2), first);
    store<i32>(kinfo + (<usize>(2 * b + 1) << 2), last);
  }
}

@inline function seqRows(rows: i32, b: i32, L: i32): i32 { return min(L, rows - b * L); }

// Masked softmax over keys first .. last of one score row p (in place, f32). Keys with mask 0
// and keys outside the window get exactly 0 (attsoftmax.wgsl); qmask false zeroes the row.
// dense (R8 hc10): every key first .. first+n-1 is valid and no window applies; the row then runs
// on four lanes with vexp.
// scale multiplies the raw scores first (R8 hc11: Q.K^T comes unscaled from the tile matmul).
function softmaxRow(p: usize, n: i32, mask: usize, first: i32, il: i32, window: i32, qmask: bool, dense: bool,
  scale: f32): void {
  if (!qmask) {
    memory.fill(p, 0, <usize>n << 2);
    return;
  }
  if (dense) {
    const vsc = f32x4.splat(scale);
    let vm = f32x4.splat(LOWEST);
    let j = 0;
    for (; j + 4 <= n; j += 4) {
      const x = f32x4.mul(v128.load(off(p, j)), vsc);
      v128.store(off(p, j), x);
      vm = f32x4.max(vm, x);
    }
    let m = Mathf.max(Mathf.max(f32x4.extract_lane(vm, 0), f32x4.extract_lane(vm, 1)),
      Mathf.max(f32x4.extract_lane(vm, 2), f32x4.extract_lane(vm, 3)));
    for (; j < n; j += 1) {
      const x = ldf(p, j) * scale;
      stf(p, j, x);
      m = Mathf.max(m, x);
    }
    const vmx = f32x4.splat(m);
    let vs = f32x4.splat(0);
    j = 0;
    for (; j + 4 <= n; j += 4) {
      const e = vexp(f32x4.sub(v128.load(off(p, j)), vmx));
      v128.store(off(p, j), e);
      vs = f32x4.add(vs, e);
    }
    let s = hsum(vs);
    for (; j < n; j += 1) {
      const e = vexp1(ldf(p, j) - m);
      stf(p, j, e);
      s += e;
    }
    const vi = f32x4.splat(s > 0 ? <f32>1.0 / s : <f32>0);
    j = 0;
    for (; j + 4 <= n; j += 4) v128.store(off(p, j), f32x4.mul(v128.load(off(p, j)), vi));
    for (; j < n; j += 1) stf(p, j, ldf(p, j) * f32x4.extract_lane(vi, 0));
    return;
  }
  if (scale != 1) for (let j = 0; j < n; j += 1) stf(p, j, ldf(p, j) * scale);
  let mx = LOWEST;
  for (let j = 0; j < n; j += 1) {
    if (keep(mask, first + j, il, window)) mx = Mathf.max(mx, ldf(p, j));
  }
  let sum: f32 = 0;
  for (let j = 0; j < n; j += 1) {
    let e: f32 = 0;
    if (keep(mask, first + j, il, window)) e = Mathf.exp(ldf(p, j) - mx);
    stf(p, j, e);
    sum += e;
  }
  const inv = sum > 0 ? <f32>1.0 / sum : <f32>0;
  for (let j = 0; j < n; j += 1) stf(p, j, ldf(p, j) * inv);
}

// Rows x[j ld + d] (j < n, d < D) into panels [ceil(n / 16)][D][16], lanes past n zero: the
// right-hand side of gemmP for Q.K^T and the relative terms (R8 hc11).
function packPanels(x: usize, ld: i32, n: i32, D: i32, out: usize): void {
  const nb = (n + 15) >> 4;
  for (let pb = 0; pb < nb; pb += 1) {
    const base = off(out, pb * D * 16);
    const j0 = pb * 16;
    const lanes = min(16, n - j0);
    if (lanes < 16) memory.fill(base, 0, <usize>(D * 16) << 2);
    for (let l = 0; l < lanes; l += 1) {
      const src = off(x, (j0 + l) * ld);
      for (let d = 0; d < D; d += 1) stf(base, d * 16 + l, ldf(src, d));
    }
  }
}

// true when every key first .. last of a sequence's mask is valid (the prefix masks of EncoderModel)
function allKeys(mask: usize, first: i32, last: i32): bool {
  for (let j = first; j <= last; j += 1) if (ldf(mask, j) <= 0.5) return false;
  return true;
}

@inline function keep(mask: usize, j: i32, il: i32, window: i32): bool {
  if (ldf(mask, j) <= 0.5) return false;
  return window == 0 || abs(il - j) <= window;
}

// scores[(row H + h) L + j] = SCALE q_row . k_j for the keys first .. last (attscore.wgsl).
function attscore(qkv: usize, scores: usize, kinfo: usize, L: i32, H: i32, D: i32, scale: f32,
  rows: i32, batch: i32): void {
  const hd = H * D;
  for (let b = 0; b < batch; b += 1) {
    const nq = seqRows(rows, b, L);
    const first = ldi(kinfo, 2 * b);
    const last = ldi(kinfo, 2 * b + 1);
    if (nq <= 0 || first >= L) continue;
    for (let h = 0; h < H; h += 1) {
      gemmNT(off(qkv, b * L * 3 * hd + h * D), 3 * hd,
        off(qkv, (b * L + first) * 3 * hd + (H + h) * D), 3 * hd,
        off(scores, (b * L * H + h) * L + first), H * L, 0,
        nq, last - first + 1, D, scale, 0);
    }
  }
}

function attsoftmax(mask: usize, scores: usize, kinfo: usize, L: i32, H: i32, window: i32,
  rows: i32): void {
  for (let i = 0; i < rows; i += 1) {
    const b = i / L;
    const first = ldi(kinfo, 2 * b);
    const last = ldi(kinfo, 2 * b + 1);
    if (first >= L) continue;
    const mk = off(mask, b * L);
    for (let h = 0; h < H; h += 1) {
      softmaxRow(off(scores, (i * H + h) * L + first), last - first + 1, mk, first,
        i - b * L, window, ldf(mask, i) > 0.5, false, 1);
    }
  }
}

// ctx[row][h D + d] = sum_j P[row][h][j] v_j (attpv.wgsl); no valid key gives a zero row.
function attpv(scores: usize, qkv: usize, ctx: usize, kinfo: usize, L: i32, H: i32, D: i32,
  rows: i32, batch: i32): void {
  const hd = H * D;
  for (let b = 0; b < batch; b += 1) {
    const nq = seqRows(rows, b, L);
    if (nq <= 0) continue;
    const first = ldi(kinfo, 2 * b);
    const last = ldi(kinfo, 2 * b + 1);
    if (first >= L) {
      memory.fill(off(ctx, b * L * hd), 0, <usize>(nq * hd) << 2);
      continue;
    }
    for (let h = 0; h < H; h += 1) {
      gemmNN(off(scores, (b * L * H + h) * L + first), H * L,
        off(qkv, (b * L + first) * 3 * hd + (2 * H + h) * D), 3 * hd,
        off(ctx, b * L * hd + h * D), hd, nq, D, last - first + 1);
    }
  }
}

// rel[row][h][m] = x_row . T[MOFF + m] for the positions lo .. hi the valid pairs reach; x is
// the q part (PART 0, c2p) or the k part (PART 1, p2c) of qkv (attrel.wgsl).
function attrel(qkv: usize, table: usize, relidx: usize, out: usize, kinfo: usize, H: i32, D: i32,
  NM: i32, moff: i32, part: i32, L: i32, rows: i32, batch: i32): void {
  const hd = H * D;
  for (let b = 0; b < batch; b += 1) {
    const nq = seqRows(rows, b, L);
    const first = ldi(kinfo, 2 * b);
    const last = ldi(kinfo, 2 * b + 1);
    if (nq <= 0 || first >= L) continue;
    const lo = ldi(relidx, first * L + last) - moff;
    const hi = ldi(relidx, last * L + first) - moff;
    for (let h = 0; h < H; h += 1) {
      gemmNT(off(qkv, b * L * 3 * hd + (part * H + h) * D), 3 * hd,
        off(table, (moff + lo) * hd + h * D), hd,
        off(out, (b * L * H + h) * NM + lo), H * NM, 0, nq, hi - lo + 1, D, 1, 0);
    }
  }
}

// scores += (c2p[i][h][p] + p2c[j][h][p]) * INVSCALE with p = relidx[i][j] - MOFF, then the
// masked softmax (attsoftrel.wgsl).
function attsoftrel(mask: usize, relidx: usize, c2p: usize, p2c: usize, scores: usize, kinfo: usize,
  L: i32, H: i32, NM: i32, moff: i32, invscale: f32, rows: i32): void {
  for (let i = 0; i < rows; i += 1) {
    const b = i / L;
    const il = i - b * L;
    const first = ldi(kinfo, 2 * b);
    const last = ldi(kinfo, 2 * b + 1);
    if (first >= L) continue;
    const qm = ldf(mask, i) > 0.5;
    for (let h = 0; h < H; h += 1) {
      const p = off(scores, (i * H + h) * L + first);
      if (qm) {
        const c = off(c2p, (i * H + h) * NM);
        for (let j = first; j <= last; j += 1) {
          if (ldf(mask, b * L + j) <= 0.5) continue;
          const m = ldi(relidx, il * L + j) - moff;
          stf(p, j - first, ldf(p, j - first)
            + (ldf(c, m) + ldf(p2c, ((b * L + j) * H + h) * NM + m)) * invscale);
        }
      }
      softmaxRow(p, last - first + 1, off(mask, b * L), first, il, 0, qm, false, 1);
    }
  }
}

// Fused standard attention (mbattention.wgsl, mbflash.wgsl) for sequence b, head h and the query
// rows i0 .. i1-1: the score block in scratch [i1 - i0][L], softmax, then the context rows. Every
// query row is computed alone, so any split into row blocks gives the same bits (R8 threads).
function attstdBlock(qkv: usize, mask: usize, ctx: usize, kinfo: usize, scratch: usize, L: i32, H: i32,
  D: i32, scale: f32, window: i32, rows: i32, b: i32, h: i32, i0: i32, i1: i32): void {
  const hd = H * D;
  const e = min(i1, seqRows(rows, b, L));
  if (e <= i0) return;
  const first = ldi(kinfo, 2 * b);
  const last = ldi(kinfo, 2 * b + 1);
  if (first >= L) {
    for (let i = i0; i < e; i += 1) memory.fill(off(ctx, (b * L + i) * hd + h * D), 0, <usize>D << 2);
    return;
  }
  const nk = last - first + 1;
  const dense = window == 0 && allKeys(off(mask, b * L), first, last);
  // the keys of head h in panels after the score block, then Q.K^T with the matmul tile
  const kp = off(scratch, L * L);
  packPanels(off(qkv, (b * L + first) * 3 * hd + (H + h) * D), 3 * hd, nk, D, kp);
  gemmP(off(qkv, (b * L + i0) * 3 * hd + h * D), 3 * hd, kp, scratch, L, e - i0, nk, D);
  for (let i = i0; i < e; i += 1) {
    softmaxRow(off(scratch, (i - i0) * L), nk, off(mask, b * L), first, i, window, ldf(mask, b * L + i) > 0.5, dense, scale);
  }
  gemmNN(scratch, L, off(qkv, (b * L + first) * 3 * hd + (2 * H + h) * D), 3 * hd,
    off(ctx, (b * L + i0) * hd + h * D), hd, e - i0, D, nk);
}

// Fused DeBERTa attention (attention.wgsl): score (q.k + q.posKey[p] + k.posQuery[p]) / SCALE
// with p = relidx[i][j]; scratch holds the score block [nq][L], then c2p [nq][S] and p2c [L][S]
// over the positions lo .. hi the pairs reach (S = 2 L, the table rows a bucket of L can use).
// One sequence b and head h (R8 threads split DeBERTa attention by heads).
function attrelf(qkv: usize, posKey: usize, posQuery: usize, relidx: usize, mask: usize, ctx: usize,
  kinfo: usize, scratch: usize, L: i32, H: i32, D: i32, scale: f32, rows: i32, b: i32, h: i32): void {
  const hd = H * D;
  const S = 2 * L;
  const c2p = off(scratch, L * L);
  const p2c = off(c2p, L * S);
  {
    const nq = seqRows(rows, b, L);
    if (nq <= 0) return;
    const first = ldi(kinfo, 2 * b);
    const last = ldi(kinfo, 2 * b + 1);
    if (first >= L) {
      for (let i = 0; i < nq; i += 1) memory.fill(off(ctx, (b * L + i) * hd + h * D), 0, <usize>D << 2);
      return;
    }
    const nk = last - first + 1;
    const lo = ldi(relidx, first * L + last);
    const np = ldi(relidx, last * L + first) - lo + 1;
    const q = off(qkv, b * L * 3 * hd);
    const inv = <f32>1.0 / scale;
    const dense = allKeys(off(mask, b * L), first, last);
    {
      // q.k, c2p and p2c with the matmul tile on panels packed after p2c (R8 hc11)
      const kp = off(p2c, L * S);
      packPanels(off(q, first * 3 * hd + (H + h) * D), 3 * hd, nk, D, kp);
      gemmP(off(q, h * D), 3 * hd, kp, scratch, L, nq, nk, D);
      packPanels(off(posKey, lo * hd + h * D), hd, np, D, kp);
      gemmP(off(q, h * D), 3 * hd, kp, c2p, S, nq, np, D);
      packPanels(off(posQuery, lo * hd + h * D), hd, np, D, kp);
      gemmP(off(q, first * 3 * hd + (H + h) * D), 3 * hd, kp, p2c, S, nk, np, D);
      for (let i = 0; i < nq; i += 1) {
        const p = off(scratch, i * L);
        const qm = ldf(mask, b * L + i) > 0.5;
        if (qm) {
          for (let j = first; j <= last; j += 1) {
            const m = ldi(relidx, i * L + j) - lo;
            stf(p, j - first, (ldf(p, j - first) + ldf(c2p, i * S + m) + ldf(p2c, (j - first) * S + m)) * inv);
          }
        }
        softmaxRow(p, nk, off(mask, b * L), first, i, 0, qm, dense, 1);
      }
      gemmNN(scratch, L, off(q, first * 3 * hd + (2 * H + h) * D), 3 * hd,
        off(ctx, b * L * hd + h * D), hd, nq, D, nk);
    }
  }
}

// ---- stream ------------------------------------------------------------------

@inline function ap(op: usize, i: i32): usize { return <usize>load<u32>(op + (<usize>(i + 2) << 2)); }
@inline function ai(op: usize, i: i32): i32 { return load<i32>(op + (<usize>(i + 2) << 2)); }
@inline function af(op: usize, i: i32): f32 { return load<f32>(op + (<usize>(i + 2) << 2)); }
@inline function opAt(stream: usize, o: i32): usize { return stream + (<usize>(HEAD + o * OPW) << 2); }
@inline function rowsOf(op: usize, rows: i32): i32 { const f = ldi(op, 1); return f >= 0 ? f : rows; }
// this thread's scratch block: header word 5 is the base, word 6 the bytes per thread
@inline function scratchOf(stream: usize, tid: i32): usize {
  return <usize>load<u32>(stream + 20) + <usize>tid * <usize>load<u32>(stream + 24);
}

// One forward over `rows` rows (the real rows of a B = 1 call, all rows of a batch plan).
export function run(stream: usize, rows: i32): void {
  runRange(stream, rows, 0, ldi(stream, 0));
}

// Ops from .. to-1 of the stream (R8 op profile: the host times one op per call). The key ranges
// come from the mask at op 0, so a range that starts later needs an earlier call from 0. With
// threads (header word 7, the control block of plan-mt.wasm) an op with enough work goes out in
// items to the helpers (R8 Festlegung 6); the others run here.
export function runRange(stream: usize, rows: i32, from: i32, to: i32): void {
  const ctl = <usize>load<u32>(stream + 28);
  if (from == 0) keyRanges(<usize>load<u32>(stream + 12), <usize>load<u32>(stream + 16), ldi(stream, 1), ldi(stream, 2));
  for (let o = from; o < to; o += 1) {
    const op = opAt(stream, o);
    const n = rowsOf(op, rows);
    const items = ctl ? itemsOf(stream, ctl, op, n) : 0;
    if (items > 1) parallel(stream, ctl, op, n, items);
    else serial(stream, op, n);
  }
}

function serial(stream: usize, op: usize, n: i32): void {
  const L = ldi(stream, 1);
  const batch = ldi(stream, 2);
  const kinfo = <usize>load<u32>(stream + 16);
  const scratch = scratchOf(stream, 0);
  const code = ldi(op, 0);
  if (code == GEMM_P || code == GEMM_NT) {
    // a, w, bias, c, N, K, act, M
    const N = ai(op, 4);
    const K = ai(op, 5);
    const m = min(n, ai(op, 7));
    if (code == GEMM_P) {
      gemmP(ap(op, 0), K, ap(op, 1), ap(op, 3), N, m, N, K);
      if (ap(op, 2) || ai(op, 6)) biasAct(ap(op, 3), N, ap(op, 2), m, N, ai(op, 6));
    } else gemmNT(ap(op, 0), K, ap(op, 1), K, ap(op, 3), N, ap(op, 2), m, N, K, 1, ai(op, 6));
  } else if (code == LN) {
    layernorm(ap(op, 0), ap(op, 1), ap(op, 2), ap(op, 3), ap(op, 4), ap(op, 5),
      ai(op, 6), ai(op, 7), af(op, 8), 0, n);
  } else if (code == EMBLN) {
    embln(ap(op, 0), ap(op, 1), ap(op, 2), ap(op, 3), ap(op, 4), ap(op, 5), ap(op, 6), ap(op, 7),
      ai(op, 8), ai(op, 9), ai(op, 10), ai(op, 11), af(op, 12), ai(op, 13), ai(op, 14), n);
  } else if (code == ADD) {
    add(ap(op, 0), ap(op, 1), min(ai(op, 2), n * ai(op, 3)), ai(op, 3), ai(op, 4), ai(op, 5));
  } else if (code == GATHER) {
    gather(ap(op, 0), ap(op, 1), ap(op, 2), ai(op, 3), ai(op, 4), ai(op, 5), ai(op, 6));
  } else if (code == POOL) {
    pool(ap(op, 0), ap(op, 1), ap(op, 2), ai(op, 3), ai(op, 4), ai(op, 5), ai(op, 6));
  } else if (code == GEGLU) {
    geglu(ap(op, 0), ap(op, 1), ai(op, 2), 0, n);
  } else if (code == ROPE) {
    rope(ap(op, 0), ap(op, 1), ai(op, 2), ai(op, 3), ai(op, 4), n);
  } else if (code == IM2COL) {
    im2col(ap(op, 0), ap(op, 1), ap(op, 2), ai(op, 3), ai(op, 4), ai(op, 5), n);
  } else if (code == ATTSCORE) {
    attscore(ap(op, 0), ap(op, 1), kinfo, ai(op, 2), ai(op, 3), ai(op, 4), af(op, 5), n, batch);
  } else if (code == ATTSOFTMAX) {
    attsoftmax(ap(op, 0), ap(op, 1), kinfo, ai(op, 2), ai(op, 3), ai(op, 4), n);
  } else if (code == ATTPV) {
    attpv(ap(op, 0), ap(op, 1), ap(op, 2), kinfo, ai(op, 3), ai(op, 4), ai(op, 5), n, batch);
  } else if (code == ATTREL) {
    attrel(ap(op, 0), ap(op, 1), ap(op, 2), ap(op, 3), kinfo, ai(op, 4), ai(op, 5), ai(op, 6),
      ai(op, 7), ai(op, 8), ai(op, 9), n, batch);
  } else if (code == ATTSOFTREL) {
    attsoftrel(ap(op, 0), ap(op, 1), ap(op, 2), ap(op, 3), ap(op, 4), kinfo, ai(op, 5), ai(op, 6),
      ai(op, 7), ai(op, 8), af(op, 9), n);
  } else if (code == ATTSTD) {
    const Lo = ai(op, 3);
    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < ai(op, 4); h += 1) {
        attstdBlock(ap(op, 0), ap(op, 1), ap(op, 2), kinfo, scratch, Lo, ai(op, 4), ai(op, 5),
          af(op, 6), ai(op, 7), n, b, h, 0, Lo);
      }
    }
  } else if (code == ATTRELF) {
    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < ai(op, 7); h += 1) {
        attrelf(ap(op, 0), ap(op, 1), ap(op, 2), ap(op, 3), ap(op, 4), ap(op, 5), kinfo, scratch,
          ai(op, 6), ai(op, 7), ai(op, 8), af(op, 9), n, b, h);
      }
    }
  } else if (code == MASKLOGITS) {
    masklogits(ap(op, 0), ap(op, 1), ap(op, 2), ai(op, 3), af(op, 4), ai(op, 5));
  } else {
    unreachable();
  }
}

// ---- threads (plan-mt.wasm, R8 Festlegung 6) ------------------------------------
//
// Control block of CTL_WORDS i32 at ctl, shared by the coordinator (the thread that calls run)
// and the helpers (helper(ctl, tid) in their own workers, same module and memory):
//   gen: bumped per dispatched op; op, rows, items, stream: the op; next: the item counter;
//   pending: helpers that have not finished the op; stop; sleepers; threads; rb, cb: rows and
//   columns per matmul item (multiples of 4 and 16), qb: query rows per attention item.
// Every output element is computed by one thread with the same code and k order as the serial
// path, so a forward gives the same bits at every thread count.

const C_GEN = 0; const C_OP = 1; const C_ROWS = 2; const C_ITEMS = 3; const C_NEXT = 4;
const C_PENDING = 5; const C_STOP = 6; const C_SLEEP = 7; const C_THREADS = 8; const C_RB = 9;
const C_CB = 10; const C_QB = 11; const C_STREAM = 12; const C_MINWORK = 13;
// spins of an idle helper before it sleeps on gen (about 0.1 to 0.3 ms)
const SPIN: i32 = 1 << 16;

@inline function cw(ctl: usize, i: i32): usize { return ctl + (<usize>i << 2); }
@inline function cdiv(a: i32, b: i32): i32 { return (a + b - 1) / b; }

// Work items of an op, 0 or 1 for an op the coordinator runs alone.
function itemsOf(stream: usize, ctl: usize, op: usize, n: i32): i32 {
  const code = ldi(op, 0);
  const minWork = load<i32>(cw(ctl, C_MINWORK));
  if (code == GEMM_P) {
    const m = min(n, ai(op, 7));
    const N = ai(op, 4);
    if (<i64>m * <i64>N * <i64>ai(op, 5) < <i64>minWork) return 0;
    return cdiv(m, load<i32>(cw(ctl, C_RB))) * cdiv(N, load<i32>(cw(ctl, C_CB)));
  }
  if (code == LN) return n >= 64 ? cdiv(n, 16) : 0;
  if (code == GEGLU) return n >= 16 ? cdiv(n, 8) : 0;
  if (code == ATTSTD) {
    const batch = ldi(stream, 2);
    const L = ai(op, 3);
    return batch * ai(op, 4) * cdiv(batch == 1 ? min(n, L) : L, load<i32>(cw(ctl, C_QB)));
  }
  if (code == ATTRELF) return ldi(stream, 2) * ai(op, 7);
  return 0;
}

// Item i of an op on thread tid.
function work(stream: usize, ctl: usize, op: usize, n: i32, i: i32, tid: i32): void {
  const code = ldi(op, 0);
  if (code == GEMM_P) {
    const N = ai(op, 4);
    const K = ai(op, 5);
    const m = min(n, ai(op, 7));
    const rb = load<i32>(cw(ctl, C_RB));
    const cb = load<i32>(cw(ctl, C_CB));
    const nc = cdiv(N, cb);
    const r0 = (i / nc) * rb;
    const c0 = (i % nc) * cb;
    const rr = min(rb, m - r0);
    const cc = min(cb, N - c0);
    const c = ap(op, 3) + (<usize>(r0 * N + c0) << 2);
    gemmP(ap(op, 0) + (<usize>(r0 * K) << 2), K, ap(op, 1) + <usize>(c0 >> 4) * (<usize>K << 6), c, N, rr, cc, K);
    const bias = ap(op, 2);
    if (bias || ai(op, 6)) biasAct(c, N, bias ? bias + (<usize>c0 << 2) : 0, rr, cc, ai(op, 6));
  } else if (code == LN) {
    layernorm(ap(op, 0), ap(op, 1), ap(op, 2), ap(op, 3), ap(op, 4), ap(op, 5),
      ai(op, 6), ai(op, 7), af(op, 8), i * 16, min(n, i * 16 + 16));
  } else if (code == GEGLU) {
    geglu(ap(op, 0), ap(op, 1), ai(op, 2), i * 8, min(n, i * 8 + 8));
  } else if (code == ATTSTD) {
    const batch = ldi(stream, 2);
    const L = ai(op, 3);
    const H = ai(op, 4);
    const qb = load<i32>(cw(ctl, C_QB));
    const nb = cdiv(batch == 1 ? min(n, L) : L, qb);
    const b = i / (H * nb);
    const h = (i / nb) % H;
    const q0 = (i % nb) * qb;
    attstdBlock(ap(op, 0), ap(op, 1), ap(op, 2), <usize>load<u32>(stream + 16), scratchOf(stream, tid),
      L, H, ai(op, 5), af(op, 6), ai(op, 7), n, b, h, q0, q0 + qb);
  } else if (code == ATTRELF) {
    const H = ai(op, 7);
    attrelf(ap(op, 0), ap(op, 1), ap(op, 2), ap(op, 3), ap(op, 4), ap(op, 5), <usize>load<u32>(stream + 16),
      scratchOf(stream, tid), ai(op, 6), H, ai(op, 8), af(op, 9), n, i / H, i % H);
  }
}

function claim(stream: usize, ctl: usize, op: usize, n: i32, items: i32, tid: i32): void {
  if (ASC_FEATURE_THREADS) {
    for (;;) {
      const i = atomic.add<i32>(cw(ctl, C_NEXT), 1);
      if (i >= items) break;
      work(stream, ctl, op, n, i, tid);
    }
  }
}

function parallel(stream: usize, ctl: usize, op: usize, n: i32, items: i32): void {
  if (ASC_FEATURE_THREADS) {
    const helpers = load<i32>(cw(ctl, C_THREADS)) - 1;
    store<u32>(cw(ctl, C_STREAM), <u32>stream);
    store<u32>(cw(ctl, C_OP), <u32>op);
    store<i32>(cw(ctl, C_ROWS), n);
    store<i32>(cw(ctl, C_ITEMS), items);
    atomic.store<i32>(cw(ctl, C_NEXT), 0);
    atomic.store<i32>(cw(ctl, C_PENDING), helpers);
    atomic.add<i32>(cw(ctl, C_GEN), 1);
    if (atomic.load<i32>(cw(ctl, C_SLEEP)) > 0) atomic.notify(cw(ctl, C_GEN), helpers);
    claim(stream, ctl, op, n, items, 0);
    while (atomic.load<i32>(cw(ctl, C_PENDING)) > 0) { /* spin: the helpers finish their items */ }
  } else {
    serial(stream, op, n);
  }
}

// Body of a helper worker: waits for each dispatched op, takes items until none are left,
// reports, until stop. Returns only on stop.
export function helper(ctl: usize, tid: i32): void {
  if (ASC_FEATURE_THREADS) {
    // gen starts at 0 with the control block; a helper that starts after the first dispatch
    // still owes that op its report, so it compares against 0, not against the current gen.
    let seen = 0;
    for (;;) {
      let spins = 0;
      while (atomic.load<i32>(cw(ctl, C_GEN)) == seen) {
        if (atomic.load<i32>(cw(ctl, C_STOP))) return;
        spins += 1;
        if (spins >= SPIN) {
          atomic.add<i32>(cw(ctl, C_SLEEP), 1);
          atomic.wait<i32>(cw(ctl, C_GEN), seen, -1);
          atomic.sub<i32>(cw(ctl, C_SLEEP), 1);
          spins = 0;
        }
      }
      seen = atomic.load<i32>(cw(ctl, C_GEN));
      if (atomic.load<i32>(cw(ctl, C_STOP))) return;
      claim(<usize>load<u32>(cw(ctl, C_STREAM)), ctl, <usize>load<u32>(cw(ctl, C_OP)),
        load<i32>(cw(ctl, C_ROWS)), load<i32>(cw(ctl, C_ITEMS)), tid);
      atomic.sub<i32>(cw(ctl, C_PENDING), 1);
    }
  }
}

// Wakes the helpers so they see stop (the host sets stop first).
export function wake(ctl: usize): void {
  if (ASC_FEATURE_THREADS) {
    atomic.add<i32>(cw(ctl, C_GEN), 1);
    atomic.notify(cw(ctl, C_GEN), -1);
  }
}
