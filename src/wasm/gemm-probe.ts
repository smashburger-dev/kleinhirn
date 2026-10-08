// R8 matmul micro-bench (docs/R8_WORKORDER.md Phase 0 step 3, Phase 1): variants of the f32
// matmul C[M, N] = A[M, K] B of src/wasm/plan.ts, each its own exported function, timed by
// bench/r8-gemm.ts in every browser. Not part of the bundle. Two builds: gemm-probe.wasm (madd is
// mul then add) and gemm-probe-relaxed.wasm (madd is f32x4.relaxed_madd).
//   v0  gemmNN of plan.ts as of R2: 4 x 16, row checks inside the k loop, B [K, N]
//   v1  4 x 16 without branches in the k loop (full 4-row tiles, then 1-row tiles), B [K, N]
//   v2  4 x 8, B [K, N]
//   v3  4 x 16 on panels: B packed [N / 16][K][16]
//   v4  8 x 8 on panels [N / 8][K][8]
//   v5  6 x 16 on panels [N / 16][K][16]
//   v6  8 x 16 on panels [N / 16][K][16]
//   peak  16 independent accumulators in registers, no memory: the multiply-add throughput
// N must be a multiple of 16 (8 for v2, v4); the bench shapes are.

export function heapBase(): usize {
  return __heap_base;
}

@inline function madd(a: v128, b: v128, c: v128): v128 {
  if (ASC_FEATURE_RELAXED_SIMD) return f32x4.relaxed_madd(a, b, c);
  return f32x4.add(f32x4.mul(a, b), c);
}

@inline function sp(p: usize): v128 {
  return f32x4.splat(load<f32>(p));
}

export function v0(a: usize, b: usize, c: usize, M: i32, N: i32, K: i32): void {
  const sa = <usize>K << 2;
  for (let mb = 0; mb < M; mb += 4) {
    const rows = min(M - mb, 4);
    const aRow = a + (<usize>(mb * K) << 2);
    for (let nb = 0; nb < N; nb += 16) {
      let a00 = f32x4.splat(0); let a01 = a00; let a02 = a00; let a03 = a00;
      let a10 = a00; let a11 = a00; let a12 = a00; let a13 = a00;
      let a20 = a00; let a21 = a00; let a22 = a00; let a23 = a00;
      let a30 = a00; let a31 = a00; let a32 = a00; let a33 = a00;
      let bp = b + (<usize>nb << 2);
      const sb = <usize>N << 2;
      for (let k = 0; k < K; k += 1) {
        const w0 = v128.load(bp);
        const w1 = v128.load(bp, 16);
        const w2 = v128.load(bp, 32);
        const w3 = v128.load(bp, 48);
        bp += sb;
        const ap = aRow + (<usize>k << 2);
        const x0 = f32x4.splat(load<f32>(ap));
        a00 = f32x4.add(a00, f32x4.mul(x0, w0));
        a01 = f32x4.add(a01, f32x4.mul(x0, w1));
        a02 = f32x4.add(a02, f32x4.mul(x0, w2));
        a03 = f32x4.add(a03, f32x4.mul(x0, w3));
        if (rows >= 2) {
          const x1 = f32x4.splat(load<f32>(ap + sa));
          a10 = f32x4.add(a10, f32x4.mul(x1, w0));
          a11 = f32x4.add(a11, f32x4.mul(x1, w1));
          a12 = f32x4.add(a12, f32x4.mul(x1, w2));
          a13 = f32x4.add(a13, f32x4.mul(x1, w3));
          if (rows >= 3) {
            const x2 = f32x4.splat(load<f32>(ap + 2 * sa));
            a20 = f32x4.add(a20, f32x4.mul(x2, w0));
            a21 = f32x4.add(a21, f32x4.mul(x2, w1));
            a22 = f32x4.add(a22, f32x4.mul(x2, w2));
            a23 = f32x4.add(a23, f32x4.mul(x2, w3));
            if (rows >= 4) {
              const x3 = f32x4.splat(load<f32>(ap + 3 * sa));
              a30 = f32x4.add(a30, f32x4.mul(x3, w0));
              a31 = f32x4.add(a31, f32x4.mul(x3, w1));
              a32 = f32x4.add(a32, f32x4.mul(x3, w2));
              a33 = f32x4.add(a33, f32x4.mul(x3, w3));
            }
          }
        }
      }
      const sc = <usize>N << 2;
      const c0 = c + (<usize>(mb * N + nb) << 2);
      v128.store(c0, a00); v128.store(c0, a01, 16); v128.store(c0, a02, 32); v128.store(c0, a03, 48);
      if (rows >= 2) { const c1 = c0 + sc; v128.store(c1, a10); v128.store(c1, a11, 16); v128.store(c1, a12, 32); v128.store(c1, a13, 48); }
      if (rows >= 3) { const c2 = c0 + 2 * sc; v128.store(c2, a20); v128.store(c2, a21, 16); v128.store(c2, a22, 32); v128.store(c2, a23, 48); }
      if (rows >= 4) { const c3 = c0 + 3 * sc; v128.store(c3, a30); v128.store(c3, a31, 16); v128.store(c3, a32, 32); v128.store(c3, a33, 48); }
    }
  }
}

// One row times 16 columns, B with row stride sb bytes (rows of B 16 floats apart in a panel).
@inline function row16(ap: usize, bp: usize, sb: usize, c: usize, K: i32): void {
  let s0 = f32x4.splat(0); let s1 = s0; let s2 = s0; let s3 = s0;
  for (let k = 0; k < K; k += 1) {
    const x = sp(ap + (<usize>k << 2));
    s0 = madd(x, v128.load(bp), s0);
    s1 = madd(x, v128.load(bp, 16), s1);
    s2 = madd(x, v128.load(bp, 32), s2);
    s3 = madd(x, v128.load(bp, 48), s3);
    bp += sb;
  }
  v128.store(c, s0); v128.store(c, s1, 16); v128.store(c, s2, 32); v128.store(c, s3, 48);
}

// 4 x 16 tile, B rows sb bytes apart; panels (sb = 64) or [K, N] (sb = 4 N).
@inline function tile4x16(ap: usize, sa: usize, bp: usize, sb: usize, c: usize, sc: usize, K: i32): void {
  let a00 = f32x4.splat(0); let a01 = a00; let a02 = a00; let a03 = a00;
  let a10 = a00; let a11 = a00; let a12 = a00; let a13 = a00;
  let a20 = a00; let a21 = a00; let a22 = a00; let a23 = a00;
  let a30 = a00; let a31 = a00; let a32 = a00; let a33 = a00;
  for (let k = 0; k < K; k += 1) {
    const w0 = v128.load(bp);
    const w1 = v128.load(bp, 16);
    const w2 = v128.load(bp, 32);
    const w3 = v128.load(bp, 48);
    bp += sb;
    const p = ap + (<usize>k << 2);
    const x0 = sp(p);
    a00 = madd(x0, w0, a00); a01 = madd(x0, w1, a01); a02 = madd(x0, w2, a02); a03 = madd(x0, w3, a03);
    const x1 = sp(p + sa);
    a10 = madd(x1, w0, a10); a11 = madd(x1, w1, a11); a12 = madd(x1, w2, a12); a13 = madd(x1, w3, a13);
    const x2 = sp(p + 2 * sa);
    a20 = madd(x2, w0, a20); a21 = madd(x2, w1, a21); a22 = madd(x2, w2, a22); a23 = madd(x2, w3, a23);
    const x3 = sp(p + 3 * sa);
    a30 = madd(x3, w0, a30); a31 = madd(x3, w1, a31); a32 = madd(x3, w2, a32); a33 = madd(x3, w3, a33);
  }
  v128.store(c, a00); v128.store(c, a01, 16); v128.store(c, a02, 32); v128.store(c, a03, 48);
  c += sc; v128.store(c, a10); v128.store(c, a11, 16); v128.store(c, a12, 32); v128.store(c, a13, 48);
  c += sc; v128.store(c, a20); v128.store(c, a21, 16); v128.store(c, a22, 32); v128.store(c, a23, 48);
  c += sc; v128.store(c, a30); v128.store(c, a31, 16); v128.store(c, a32, 32); v128.store(c, a33, 48);
}

export function v1(a: usize, b: usize, c: usize, M: i32, N: i32, K: i32): void {
  const sa = <usize>K << 2;
  const sb = <usize>N << 2;
  const sc = <usize>N << 2;
  let m = 0;
  for (; m + 4 <= M; m += 4) {
    for (let nb = 0; nb < N; nb += 16) {
      tile4x16(a + <usize>m * sa, sa, b + (<usize>nb << 2), sb, c + <usize>m * sc + (<usize>nb << 2), sc, K);
    }
  }
  for (; m < M; m += 1) {
    for (let nb = 0; nb < N; nb += 16) row16(a + <usize>m * sa, b + (<usize>nb << 2), sb, c + <usize>m * sc + (<usize>nb << 2), K);
  }
}

export function v2(a: usize, b: usize, c: usize, M: i32, N: i32, K: i32): void {
  const sa = <usize>K << 2;
  const sb = <usize>N << 2;
  const sc = <usize>N << 2;
  for (let m = 0; m < M; m += 4) {
    for (let nb = 0; nb < N; nb += 8) {
      let a00 = f32x4.splat(0); let a01 = a00; let a10 = a00; let a11 = a00;
      let a20 = a00; let a21 = a00; let a30 = a00; let a31 = a00;
      let bp = b + (<usize>nb << 2);
      const ap = a + <usize>m * sa;
      for (let k = 0; k < K; k += 1) {
        const w0 = v128.load(bp);
        const w1 = v128.load(bp, 16);
        bp += sb;
        const p = ap + (<usize>k << 2);
        const x0 = sp(p); a00 = madd(x0, w0, a00); a01 = madd(x0, w1, a01);
        const x1 = sp(p + sa); a10 = madd(x1, w0, a10); a11 = madd(x1, w1, a11);
        const x2 = sp(p + 2 * sa); a20 = madd(x2, w0, a20); a21 = madd(x2, w1, a21);
        const x3 = sp(p + 3 * sa); a30 = madd(x3, w0, a30); a31 = madd(x3, w1, a31);
      }
      let o = c + <usize>m * sc + (<usize>nb << 2);
      v128.store(o, a00); v128.store(o, a01, 16); o += sc;
      v128.store(o, a10); v128.store(o, a11, 16); o += sc;
      v128.store(o, a20); v128.store(o, a21, 16); o += sc;
      v128.store(o, a30); v128.store(o, a31, 16);
    }
  }
}

// Panels: p16 holds B as [N / 16][K][16], p8 as [N / 8][K][8].
export function v3(a: usize, p16: usize, c: usize, M: i32, N: i32, K: i32): void {
  const sa = <usize>K << 2;
  const sc = <usize>N << 2;
  const ps = <usize>K << 6; // bytes per panel
  let m = 0;
  for (; m + 4 <= M; m += 4) {
    for (let nb = 0; nb < N; nb += 16) {
      tile4x16(a + <usize>m * sa, sa, p16 + <usize>(nb >> 4) * ps, 64, c + <usize>m * sc + (<usize>nb << 2), sc, K);
    }
  }
  for (; m < M; m += 1) {
    for (let nb = 0; nb < N; nb += 16) row16(a + <usize>m * sa, p16 + <usize>(nb >> 4) * ps, 64, c + <usize>m * sc + (<usize>nb << 2), K);
  }
}

export function v4(a: usize, p8: usize, c: usize, M: i32, N: i32, K: i32): void {
  const sa = <usize>K << 2;
  const sc = <usize>N << 2;
  const ps = <usize>K << 5;
  for (let m = 0; m < M; m += 8) {
    for (let nb = 0; nb < N; nb += 8) {
      let a00 = f32x4.splat(0); let a01 = a00; let a10 = a00; let a11 = a00;
      let a20 = a00; let a21 = a00; let a30 = a00; let a31 = a00;
      let a40 = a00; let a41 = a00; let a50 = a00; let a51 = a00;
      let a60 = a00; let a61 = a00; let a70 = a00; let a71 = a00;
      let bp = p8 + <usize>(nb >> 3) * ps;
      const ap = a + <usize>m * sa;
      for (let k = 0; k < K; k += 1) {
        const w0 = v128.load(bp);
        const w1 = v128.load(bp, 16);
        bp += 32;
        const p = ap + (<usize>k << 2);
        const x0 = sp(p); a00 = madd(x0, w0, a00); a01 = madd(x0, w1, a01);
        const x1 = sp(p + sa); a10 = madd(x1, w0, a10); a11 = madd(x1, w1, a11);
        const x2 = sp(p + 2 * sa); a20 = madd(x2, w0, a20); a21 = madd(x2, w1, a21);
        const x3 = sp(p + 3 * sa); a30 = madd(x3, w0, a30); a31 = madd(x3, w1, a31);
        const x4 = sp(p + 4 * sa); a40 = madd(x4, w0, a40); a41 = madd(x4, w1, a41);
        const x5 = sp(p + 5 * sa); a50 = madd(x5, w0, a50); a51 = madd(x5, w1, a51);
        const x6 = sp(p + 6 * sa); a60 = madd(x6, w0, a60); a61 = madd(x6, w1, a61);
        const x7 = sp(p + 7 * sa); a70 = madd(x7, w0, a70); a71 = madd(x7, w1, a71);
      }
      let o = c + <usize>m * sc + (<usize>nb << 2);
      v128.store(o, a00); v128.store(o, a01, 16); o += sc;
      v128.store(o, a10); v128.store(o, a11, 16); o += sc;
      v128.store(o, a20); v128.store(o, a21, 16); o += sc;
      v128.store(o, a30); v128.store(o, a31, 16); o += sc;
      v128.store(o, a40); v128.store(o, a41, 16); o += sc;
      v128.store(o, a50); v128.store(o, a51, 16); o += sc;
      v128.store(o, a60); v128.store(o, a61, 16); o += sc;
      v128.store(o, a70); v128.store(o, a71, 16);
    }
  }
}

export function v5(a: usize, p16: usize, c: usize, M: i32, N: i32, K: i32): void {
  const sa = <usize>K << 2;
  const sc = <usize>N << 2;
  const ps = <usize>K << 6;
  let m = 0;
  for (; m + 6 <= M; m += 6) {
    for (let nb = 0; nb < N; nb += 16) {
      let a00 = f32x4.splat(0); let a01 = a00; let a02 = a00; let a03 = a00;
      let a10 = a00; let a11 = a00; let a12 = a00; let a13 = a00;
      let a20 = a00; let a21 = a00; let a22 = a00; let a23 = a00;
      let a30 = a00; let a31 = a00; let a32 = a00; let a33 = a00;
      let a40 = a00; let a41 = a00; let a42 = a00; let a43 = a00;
      let a50 = a00; let a51 = a00; let a52 = a00; let a53 = a00;
      let bp = p16 + <usize>(nb >> 4) * ps;
      const ap = a + <usize>m * sa;
      for (let k = 0; k < K; k += 1) {
        const w0 = v128.load(bp);
        const w1 = v128.load(bp, 16);
        const w2 = v128.load(bp, 32);
        const w3 = v128.load(bp, 48);
        bp += 64;
        const p = ap + (<usize>k << 2);
        const x0 = sp(p); a00 = madd(x0, w0, a00); a01 = madd(x0, w1, a01); a02 = madd(x0, w2, a02); a03 = madd(x0, w3, a03);
        const x1 = sp(p + sa); a10 = madd(x1, w0, a10); a11 = madd(x1, w1, a11); a12 = madd(x1, w2, a12); a13 = madd(x1, w3, a13);
        const x2 = sp(p + 2 * sa); a20 = madd(x2, w0, a20); a21 = madd(x2, w1, a21); a22 = madd(x2, w2, a22); a23 = madd(x2, w3, a23);
        const x3 = sp(p + 3 * sa); a30 = madd(x3, w0, a30); a31 = madd(x3, w1, a31); a32 = madd(x3, w2, a32); a33 = madd(x3, w3, a33);
        const x4 = sp(p + 4 * sa); a40 = madd(x4, w0, a40); a41 = madd(x4, w1, a41); a42 = madd(x4, w2, a42); a43 = madd(x4, w3, a43);
        const x5 = sp(p + 5 * sa); a50 = madd(x5, w0, a50); a51 = madd(x5, w1, a51); a52 = madd(x5, w2, a52); a53 = madd(x5, w3, a53);
      }
      let o = c + <usize>m * sc + (<usize>nb << 2);
      v128.store(o, a00); v128.store(o, a01, 16); v128.store(o, a02, 32); v128.store(o, a03, 48); o += sc;
      v128.store(o, a10); v128.store(o, a11, 16); v128.store(o, a12, 32); v128.store(o, a13, 48); o += sc;
      v128.store(o, a20); v128.store(o, a21, 16); v128.store(o, a22, 32); v128.store(o, a23, 48); o += sc;
      v128.store(o, a30); v128.store(o, a31, 16); v128.store(o, a32, 32); v128.store(o, a33, 48); o += sc;
      v128.store(o, a40); v128.store(o, a41, 16); v128.store(o, a42, 32); v128.store(o, a43, 48); o += sc;
      v128.store(o, a50); v128.store(o, a51, 16); v128.store(o, a52, 32); v128.store(o, a53, 48);
    }
  }
  for (; m < M; m += 1) {
    for (let nb = 0; nb < N; nb += 16) row16(a + <usize>m * sa, p16 + <usize>(nb >> 4) * ps, 64, c + <usize>m * sc + (<usize>nb << 2), K);
  }
}

export function v6(a: usize, p16: usize, c: usize, M: i32, N: i32, K: i32): void {
  const sa = <usize>K << 2;
  const sc = <usize>N << 2;
  const ps = <usize>K << 6;
  // two 4 x 16 halves share each B load: 32 accumulators
  for (let m = 0; m < M; m += 8) {
    for (let nb = 0; nb < N; nb += 16) {
      let a00 = f32x4.splat(0); let a01 = a00; let a02 = a00; let a03 = a00;
      let a10 = a00; let a11 = a00; let a12 = a00; let a13 = a00;
      let a20 = a00; let a21 = a00; let a22 = a00; let a23 = a00;
      let a30 = a00; let a31 = a00; let a32 = a00; let a33 = a00;
      let a40 = a00; let a41 = a00; let a42 = a00; let a43 = a00;
      let a50 = a00; let a51 = a00; let a52 = a00; let a53 = a00;
      let a60 = a00; let a61 = a00; let a62 = a00; let a63 = a00;
      let a70 = a00; let a71 = a00; let a72 = a00; let a73 = a00;
      let bp = p16 + <usize>(nb >> 4) * ps;
      const ap = a + <usize>m * sa;
      for (let k = 0; k < K; k += 1) {
        const w0 = v128.load(bp);
        const w1 = v128.load(bp, 16);
        const w2 = v128.load(bp, 32);
        const w3 = v128.load(bp, 48);
        bp += 64;
        const p = ap + (<usize>k << 2);
        const x0 = sp(p); a00 = madd(x0, w0, a00); a01 = madd(x0, w1, a01); a02 = madd(x0, w2, a02); a03 = madd(x0, w3, a03);
        const x1 = sp(p + sa); a10 = madd(x1, w0, a10); a11 = madd(x1, w1, a11); a12 = madd(x1, w2, a12); a13 = madd(x1, w3, a13);
        const x2 = sp(p + 2 * sa); a20 = madd(x2, w0, a20); a21 = madd(x2, w1, a21); a22 = madd(x2, w2, a22); a23 = madd(x2, w3, a23);
        const x3 = sp(p + 3 * sa); a30 = madd(x3, w0, a30); a31 = madd(x3, w1, a31); a32 = madd(x3, w2, a32); a33 = madd(x3, w3, a33);
        const x4 = sp(p + 4 * sa); a40 = madd(x4, w0, a40); a41 = madd(x4, w1, a41); a42 = madd(x4, w2, a42); a43 = madd(x4, w3, a43);
        const x5 = sp(p + 5 * sa); a50 = madd(x5, w0, a50); a51 = madd(x5, w1, a51); a52 = madd(x5, w2, a52); a53 = madd(x5, w3, a53);
        const x6 = sp(p + 6 * sa); a60 = madd(x6, w0, a60); a61 = madd(x6, w1, a61); a62 = madd(x6, w2, a62); a63 = madd(x6, w3, a63);
        const x7 = sp(p + 7 * sa); a70 = madd(x7, w0, a70); a71 = madd(x7, w1, a71); a72 = madd(x7, w2, a72); a73 = madd(x7, w3, a73);
      }
      let o = c + <usize>m * sc + (<usize>nb << 2);
      v128.store(o, a00); v128.store(o, a01, 16); v128.store(o, a02, 32); v128.store(o, a03, 48); o += sc;
      v128.store(o, a10); v128.store(o, a11, 16); v128.store(o, a12, 32); v128.store(o, a13, 48); o += sc;
      v128.store(o, a20); v128.store(o, a21, 16); v128.store(o, a22, 32); v128.store(o, a23, 48); o += sc;
      v128.store(o, a30); v128.store(o, a31, 16); v128.store(o, a32, 32); v128.store(o, a33, 48); o += sc;
      v128.store(o, a40); v128.store(o, a41, 16); v128.store(o, a42, 32); v128.store(o, a43, 48); o += sc;
      v128.store(o, a50); v128.store(o, a51, 16); v128.store(o, a52, 32); v128.store(o, a53, 48); o += sc;
      v128.store(o, a60); v128.store(o, a61, 16); v128.store(o, a62, 32); v128.store(o, a63, 48); o += sc;
      v128.store(o, a70); v128.store(o, a71, 16); v128.store(o, a72, 32); v128.store(o, a73, 48);
    }
  }
}

// n iterations of 16 independent multiply-adds on 4 lanes: 64 n MACs, no memory traffic.
export function peak(n: i32): f32 {
  const x = f32x4.splat(<f32>1.0000001);
  const y = f32x4.splat(<f32>0.9999999);
  let s0 = f32x4.splat(0); let s1 = f32x4.splat(1); let s2 = f32x4.splat(2); let s3 = f32x4.splat(3);
  let s4 = f32x4.splat(4); let s5 = f32x4.splat(5); let s6 = f32x4.splat(6); let s7 = f32x4.splat(7);
  let s8 = f32x4.splat(8); let s9 = f32x4.splat(9); let sa = f32x4.splat(10); let sb = f32x4.splat(11);
  let sc = f32x4.splat(12); let sd = f32x4.splat(13); let se = f32x4.splat(14); let sf = f32x4.splat(15);
  for (let i = 0; i < n; i += 1) {
    s0 = madd(x, y, s0); s1 = madd(x, y, s1); s2 = madd(x, y, s2); s3 = madd(x, y, s3);
    s4 = madd(x, y, s4); s5 = madd(x, y, s5); s6 = madd(x, y, s6); s7 = madd(x, y, s7);
    s8 = madd(x, y, s8); s9 = madd(x, y, s9); sa = madd(x, y, sa); sb = madd(x, y, sb);
    sc = madd(x, y, sc); sd = madd(x, y, sd); se = madd(x, y, se); sf = madd(x, y, sf);
  }
  const t = f32x4.add(f32x4.add(f32x4.add(s0, s1), f32x4.add(s2, s3)), f32x4.add(f32x4.add(s4, s5), f32x4.add(s6, s7)));
  const u = f32x4.add(f32x4.add(f32x4.add(s8, s9), f32x4.add(sa, sb)), f32x4.add(f32x4.add(sc, sd), f32x4.add(se, sf)));
  const v = f32x4.add(t, u);
  return f32x4.extract_lane(v, 0) + f32x4.extract_lane(v, 1) + f32x4.extract_lane(v, 2) + f32x4.extract_lane(v, 3);
}
