{{ENABLE}}// GeGLU epilogue for the ModernBERT FFN: mid is the raw [L, 2*I]
// projection; out[row, j] = gelu(mid[row, j]) * mid[row, I + j] for j < I.
// gelu is the exact erf form via Abramowitz-Stegun 7.1.26 (same constants
// as matmul.wgsl ACT=2). One workgroup per row.

override I: u32 = 1152u;

@group(0) @binding(0) var<storage, read> mid: array<{{F}}>;
@group(0) @binding(1) var<storage, read_write> gate: array<{{F}}>;

fn gelu(v: f32) -> f32 {
  let u = v * 0.7071067811865476;
  let t = 1.0 / (1.0 + 0.3275911 * abs(u));
  let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t;
  let e = 1.0 - p * exp(-u * u);
  return 0.5 * v * (1.0 + select(-e, e, v >= 0.0));
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let base = row * 2u * I;
  for (var j = lid.x; j < I; j += 64u) {
    gate[row * I + j] = {{F}}(
      gelu(f32(mid[base + j])) * f32(mid[base + I + j]));
  }
}
