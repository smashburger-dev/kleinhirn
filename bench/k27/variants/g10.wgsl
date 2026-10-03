{{ENABLE}}// C[M,N] = A[M,K] * W[N,K]^T + B[N], register-blocked (K27, tools/k27_mm_gen.py g10 16 16 4 32).
// Workgroup 16x16 threads, 4x4 outputs per thread, tile 64x64, K step 32; tiles as vec4 of
// the storage type in workgroup memory (A [row][k/4], W [k][col/4]). Every output sums k in order
// in f32 and adds the bias last, like matmul.wgsl. Needs K % 32 == 0.
// dispatch: 64x64 (tile rows x cols)
// ACT: 0 none, 1 relu, 2 gelu (erf, Abramowitz-Stegun 7.1.26), 3 tanh, 4 silu.

override M: u32 = 1u;
override N: u32 = 1u;
override K: u32 = 32u;
override ACT: u32 = 0u;

@group(0) @binding(0) var<storage, read> a: array<vec4<{{F}}>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<{{F}}>>;
@group(0) @binding(2) var<storage, read> bias: array<{{F}}>;
@group(0) @binding(3) var<storage, read_write> c: array<{{F}}>;

var<workgroup> sa: array<vec4<{{F}}>, 512>;
var<workgroup> sw: array<vec4<{{F}}>, 512>;

fn activate(v: f32) -> f32 {
  if (ACT == 1u) { return max(v, 0.0); }
  if (ACT == 2u) {
    // GELU = 0.5 v (1 + erf(v / sqrt(2))); erf via Abramowitz-Stegun 7.1.26
    // on u = v / sqrt(2): erf(|u|) = 1 - p(t(u)) exp(-u*u), sign follows v.
    let u = v * 0.7071067811865476;
    let t = 1.0 / (1.0 + 0.3275911 * abs(u));
    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t;
    let e = 1.0 - p * exp(-u * u);
    return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));
  }
  if (ACT == 3u) { return tanh(v); }
  if (ACT == 4u) { return v / (1.0 + exp(-v)); }
  return v;
}

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 16u + lid.x;
  let row0 = wid.y * 64u;
  let col0 = wid.x * 64u;
  let tileRow = lid.y * 4u;
  let K4 = K / 4u;
  var acc: array<vec4<f32>, 4>;
  for (var t4 = 0u; t4 < K4; t4 += 8u) {
    for (var run = tid; run < 512u; run += 256u) {
      let ar = row0 + run / 8u;
      sa[run] = select(vec4<{{F}}>(0.0), a[ar * K4 + t4 + run % 8u], ar < M);
    }
    for (var g = tid; g < 128u; g += 256u) {
      let ng = g / 8u;
      let kv = g % 8u;
      var m: array<vec4<{{F}}>, 4>;
      for (var q = 0u; q < 4u; q += 1u) {
        let wr = col0 + ng * 4u + q;
        m[q] = select(vec4<{{F}}>(0.0), w[wr * K4 + t4 + kv], wr < N);
      }
      for (var e = 0u; e < 4u; e += 1u) {
        sw[(kv * 4u + e) * 16u + ng] = vec4<{{F}}>(m[0][e], m[1][e], m[2][e], m[3][e]);
      }
    }
    workgroupBarrier();
    for (var k4 = 0u; k4 < 8u; k4 += 1u) {
      let b0 = vec4<f32>(sw[(k4 * 4u) * 16u + lid.x]);
      let b1 = vec4<f32>(sw[(k4 * 4u + 1u) * 16u + lid.x]);
      let b2 = vec4<f32>(sw[(k4 * 4u + 2u) * 16u + lid.x]);
      let b3 = vec4<f32>(sw[(k4 * 4u + 3u) * 16u + lid.x]);
      for (var i = 0u; i < 4u; i += 1u) {
        let av = vec4<f32>(sa[(tileRow + i) * 8u + k4]);
        acc[i] = b0 * av.x + acc[i];
        acc[i] = b1 * av.y + acc[i];
        acc[i] = b2 * av.z + acc[i];
        acc[i] = b3 * av.w + acc[i];
      }
    }
    workgroupBarrier();
  }
  let col = col0 + lid.x * 4u;
  for (var i = 0u; i < 4u; i += 1u) {
    let row = row0 + tileRow + i;
    if (row >= M) { continue; }
    for (var j = 0u; j < 4u; j += 1u) {
      if (col + j < N) {
        c[row * N + col + j] = {{F}}(activate(acc[i][j] + f32(bias[col + j])));
      }
    }
  }
}
