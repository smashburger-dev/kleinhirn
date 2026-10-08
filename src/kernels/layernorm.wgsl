{{ENABLE}}// LayerNorm per row, one workgroup per row, f32 accumulation.
// MODE 0: out = LN(a)
// MODE 1: out = LN(a) * mask[row]   (embedding masking)
// MODE 2: out = LN(a + b)          (residual add fused)
// mean/var over the last dimension with manifest epsilon (1e-7).

override N: u32 = 384u;
override MODE: u32 = 0u;
override EPS: f32 = 1e-7;

@group(0) @binding(0) var<storage, read> a: array<{{F}}>;
@group(0) @binding(1) var<storage, read> b: array<{{F}}>;
@group(0) @binding(2) var<storage, read> weight: array<{{F}}>;
@group(0) @binding(3) var<storage, read> bias: array<{{F}}>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<{{F}}>;

// The row is kept here between the passes: one read from storage, and the variance is the mean of
// (v - mean)^2 (never negative, exactly 0 for a constant row) instead of E[v^2] - mean^2 (review R05).
var<workgroup> xs: array<f32, N>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let base = row * N;
  var s = 0.0;
  for (var i = lid.x; i < N; i += 64u) {
    var v = f32(a[base + i]);
    if (MODE == 2u) { v += f32(b[base + i]); }
    xs[i] = v;
    s += v;
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) {
    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }
    workgroupBarrier();
  }
  let mean = red[0] / f32(N);
  workgroupBarrier();
  var sq = 0.0;
  for (var i = lid.x; i < N; i += 64u) {
    let d = xs[i] - mean;
    sq += d * d;
  }
  red[lid.x] = sq;
  workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) {
    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }
    workgroupBarrier();
  }
  let variance = red[0] / f32(N);
  let inv = 1.0 / sqrt(variance + EPS);
  for (var i = lid.x; i < N; i += 64u) {
    var y = (xs[i] - mean) * inv * f32(weight[i]) + f32(bias[i]);
    if (MODE == 1u) { y *= mask[row]; }
    out[base + i] = {{F}}(y);
  }
}
