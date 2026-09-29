{{ENABLE}}// Elementwise residual accumulation. MODE 0: dst[i] += src[i].
// MODE 1: dst[i] += src[b * N + i % N] where b = row / L is the batch
// sequence (per-row type embeddings; at B = 1 this is src[i % N]).
// One workgroup of 64 threads per 64 elements.

override TOTAL: u32 = 1u;
override N: u32 = 384u;
override MODE: u32 = 0u;
override L: u32 = 128u;

@group(0) @binding(0) var<storage, read_write> dst: array<{{F}}>;
@group(0) @binding(1) var<storage, read> src: array<{{F}}>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= TOTAL) { return; }
  let s = select(src[i], src[(i / (N * L)) * N + i % N], MODE == 1u);
  dst[i] = {{F}}(f32(dst[i]) + f32(s));
}
