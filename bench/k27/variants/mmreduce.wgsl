{{ENABLE}}// Split-K reduce: c[r][n] = act(sum_s part[s][r][n] + bias[n]).
override M: u32 = 1u;
override N: u32 = 1u;
override ACT: u32 = 0u;
override SPLIT: u32 = 4u;
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<{{F}}>;
@group(0) @binding(2) var<storage, read_write> c: array<{{F}}>;
fn activate(v: f32) -> f32 {
  if (ACT == 2u) {
    let u = v * 0.7071067811865476;
    let t = 1.0 / (1.0 + 0.3275911 * abs(u));
    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t;
    let e = 1.0 - p * exp(-u * u);
    return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));
  }
  return v;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= M * N) { return; }
  var acc = 0.0;
  for (var s = 0u; s < SPLIT; s += 1u) { acc += part[s * M * N + i]; }
  c[i] = {{F}}(activate(acc + f32(bias[i % N])));
}
