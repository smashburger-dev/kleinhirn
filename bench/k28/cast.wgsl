enable f16;
// f32 storage in -> f16 storage out. 256 threads per workgroup, 1D grid with
// 2D wrap (x <= 65535 workgroups, rows in y) so tensors beyond
// 65535 * 256 elements dispatch too. outOffset (in f16 elements) places a
// part inside a fused destination (q|k|v rows).

struct Params {
  n: u32,
  outOffset: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f16>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let i = (wid.y * nwg.x + wid.x) * 256u + lid.x;
  if (i < p.n) {
    dst[p.outOffset + i] = f16(src[i]);
  }
}
