{{ENABLE}}// RoPE (rotate-half convention) applied in place to the q and k
// sections of the qkv buffer. One workgroup of 32 threads per (position,
// head, section): thread d handles the pair (d, d + D/2), so no shared
// state is needed. cossin holds per-position tables [L, 2 * D]: cos at
// i * 2 * D + d and sin at i * 2 * D + D + d.
// out[d]      = x[d] * cos_d - x[d + half] * sin_d
// out[d + half] = x[d + half] * cos_d + x[d] * sin_d
// (cos/sin are indexed by d mod half, which is why both halves use the
// same table entries.)
// Batch (K16): i is a global row in B*L packed space; the rotary table is
// indexed by the sequence-local position i mod L.

override L: u32 = 128u;
override H: u32 = 6u;
override D: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> qkv: array<{{F}}>;
@group(0) @binding(1) var<storage, read> cossin: array<f32>;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let i = wid.x;
  let il = i - (i / L) * L;
  let h = wid.y / 2u;
  let seg = wid.y % 2u; // 0 = q, 1 = k (v keeps absolute position, no RoPE)
  let hd = H * D;
  let base = i * 3u * hd + (seg * H + h) * D;
  let half = D / 2u;
  let d = lid.x;
  let a = f32(qkv[base + d]);
  let b = f32(qkv[base + d + half]);
  let c = cossin[il * 2u * D + d];
  let s = cossin[il * 2u * D + D + d];
  qkv[base + d] = {{F}}(a * c - b * s);
  qkv[base + d + half] = {{F}}(b * c + a * s);
}
