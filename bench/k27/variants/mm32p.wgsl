{{ENABLE}}// K27 variant mm32p: 32x32 tile, 8x8 threads, 4x4 outputs each, K step 32; A and W
// read as vec4 along k (coalesced), both tiles padded to a row stride of 33 so the
// transposed W stores and the A column reads do not hit one bank. Sums k in order in f32.
// Needs K % 32 == 0.
// dispatch: 32x32

override M: u32 = 1u;
override N: u32 = 1u;
override K: u32 = 32u;
override ACT: u32 = 0u;

@group(0) @binding(0) var<storage, read> a: array<vec4<{{F}}>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<{{F}}>>;
@group(0) @binding(2) var<storage, read> bias: array<{{F}}>;
@group(0) @binding(3) var<storage, read_write> c: array<{{F}}>;

var<workgroup> sa: array<f32, 1056>; // [row][k], stride 33
var<workgroup> sw: array<f32, 1056>; // [k][col], stride 33

fn activate(v: f32) -> f32 {
  if (ACT == 1u) { return max(v, 0.0); }
  if (ACT == 2u) {
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

@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * 8u + lid.x;
  let row0 = wid.y * 32u;
  let col0 = wid.x * 32u;
  let tileRow = lid.y * 4u;
  let tileCol = lid.x * 4u;
  let K4 = K / 4u;
  var acc: array<vec4<f32>, 4>;
  for (var t4 = 0u; t4 < K4; t4 += 8u) {
    for (var i = 0u; i < 4u; i += 1u) {
      let run = tid + 64u * i;
      let r = run / 8u;
      let kv = run % 8u;
      let ar = row0 + r;
      let wr = col0 + r;
      let av = select(vec4<f32>(0.0), vec4<f32>(a[ar * K4 + t4 + kv]), ar < M);
      let wv = select(vec4<f32>(0.0), vec4<f32>(w[wr * K4 + t4 + kv]), wr < N);
      let k0 = kv * 4u;
      sa[r * 33u + k0] = av.x;
      sa[r * 33u + k0 + 1u] = av.y;
      sa[r * 33u + k0 + 2u] = av.z;
      sa[r * 33u + k0 + 3u] = av.w;
      sw[k0 * 33u + r] = wv.x;
      sw[(k0 + 1u) * 33u + r] = wv.y;
      sw[(k0 + 2u) * 33u + r] = wv.z;
      sw[(k0 + 3u) * 33u + r] = wv.w;
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k += 1u) {
      let wv = vec4<f32>(sw[k * 33u + tileCol], sw[k * 33u + tileCol + 1u],
        sw[k * 33u + tileCol + 2u], sw[k * 33u + tileCol + 3u]);
      for (var i = 0u; i < 4u; i += 1u) {
        acc[i] += sa[(tileRow + i) * 33u + k] * wv;
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i += 1u) {
    let row = row0 + tileRow + i;
    if (row >= M) { continue; }
    for (var j = 0u; j < 4u; j += 1u) {
      let col = col0 + tileCol + j;
      if (col < N) {
        c[row * N + col] = {{F}}(activate(acc[i][j] + f32(bias[col])));
      }
    }
  }
}
