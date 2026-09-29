{{ENABLE}}// C[M,N] = A[M,K] * W[N,K]^T + B[N]. Row-major storage, 16x16 tiles,
// f32 accumulation. ACT: 0 none, 1 relu, 2 gelu (erf, Abramowitz-Stegun 7.1.26).

override M: u32 = 1u;
override N: u32 = 1u;
override K: u32 = 1u;
override ACT: u32 = 0u;

@group(0) @binding(0) var<storage, read> a: array<{{F}}>;
@group(0) @binding(1) var<storage, read> w: array<{{F}}>;
@group(0) @binding(2) var<storage, read> bias: array<{{F}}>;
@group(0) @binding(3) var<storage, read_write> c: array<{{F}}>;

var<workgroup> ta: array<f32, 256>;
var<workgroup> tw: array<f32, 256>;

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
  return v;
}

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.y * 16u + lid.y;
  let col = wid.x * 16u + lid.x;
  var acc = 0.0;
  for (var t = 0u; t < K; t += 16u) {
    let ai = t + lid.x;
    let wi = t + lid.y;
    ta[lid.y * 16u + lid.x] = select(0.0, f32(a[row * K + ai]), row < M && ai < K);
    tw[lid.y * 16u + lid.x] = select(0.0, f32(w[col * K + wi]), col < N && wi < K);
    workgroupBarrier();
    for (var k = 0u; k < 16u; k += 1u) {
      acc += ta[lid.y * 16u + k] * tw[k * 16u + lid.x];
    }
    workgroupBarrier();
  }
  if (row < M && col < N) {
    c[row * N + col] = {{F}}(activate(acc + f32(bias[col])));
  }
}
