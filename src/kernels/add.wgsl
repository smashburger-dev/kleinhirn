{{ENABLE}}// Elementwise residual accumulation. MODE 0: dst[i] += src[i].
// MODE 1: dst[i] += src[b * N + i % N] where b = row / L is the batch
// sequence (per-row type embeddings; at B = 1 this is src[i % N]).
// Workgroups of 64 threads stride over the elements (grid-stride loop), so the dispatch can be
// capped at 65535 workgroups in x whatever TOTAL is.

override TOTAL: u32 = 1u;
override N: u32 = 384u;
override MODE: u32 = 0u;
override L: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> dst: array<{{F}}>;
@group(0) @binding(1) var<storage, read> src: array<{{F}}>;

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  for (var i = gid.x; i < TOTAL; i += 64u * nwg.x) {
    let s = select(src[i], src[(i / (N * L)) * N + i % N], MODE == 1u);
    dst[i] = {{F}}(f32(dst[i]) + f32(s));
  }
}
