{{ENABLE}}// K27 variant mmv4h: ORT's loop form. 32x32 tile, 8x8 threads, 4x4 outputs each, K step 32.
// A tile [row][k/4] and W tile [k][col/4] as vec4 in workgroup memory (storage type); W rows are
// transposed in registers, 4 rows per thread. Every output sums k in order in f32.
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

var<workgroup> sa: array<vec4<{{F}}>, 256>; // [row][k4]
var<workgroup> sw: array<vec4<{{F}}>, 256>; // [k][col4]

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
  let K4 = K / 4u;
  // W loads: thread tid takes rows 4 * (tid / 8) .. + 3 at k chunk tid % 8.
  let ng = tid / 8u;
  let wkv = tid % 8u;
  var acc: array<vec4<f32>, 4>;
  for (var t4 = 0u; t4 < K4; t4 += 8u) {
    for (var i = 0u; i < 4u; i += 1u) {
      let run = tid + 64u * i;
      let ar = row0 + run / 8u;
      sa[run] = select(vec4<{{F}}>(0.0), vec4<{{F}}>(a[ar * K4 + t4 + run % 8u]), ar < M);
    }
    var m: array<vec4<{{F}}>, 4>;
    for (var q = 0u; q < 4u; q += 1u) {
      let wr = col0 + ng * 4u + q;
      m[q] = select(vec4<{{F}}>(0.0), vec4<{{F}}>(w[wr * K4 + t4 + wkv]), wr < N);
    }
    for (var e = 0u; e < 4u; e += 1u) {
      sw[(wkv * 4u + e) * 8u + ng] = vec4<{{F}}>(m[0][e], m[1][e], m[2][e], m[3][e]);
    }
    workgroupBarrier();
    for (var k4 = 0u; k4 < 8u; k4 += 1u) {
      let b0 = vec4<f32>(sw[(k4 * 4u) * 8u + lid.x]);
      let b1 = vec4<f32>(sw[(k4 * 4u + 1u) * 8u + lid.x]);
      let b2 = vec4<f32>(sw[(k4 * 4u + 2u) * 8u + lid.x]);
      let b3 = vec4<f32>(sw[(k4 * 4u + 3u) * 8u + lid.x]);
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
