{{ENABLE}}// states[g, :] = x[markers[g], :] — collect hidden states at the
// classification marker positions. One workgroup, threads stride over rows*D.
// Batch (K16): markers hold B blocks of 3*K (indices, mask bits, groups);
// output row g belongs to sequence b = g / K and reads x at the
// sequence-local marker index plus b * L.

override K: u32 = 16u;
override D: u32 = 384u;
override L: u32 = 128u;

@group(0) @binding(0) var<storage, read> markers: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<{{F}}>;
@group(0) @binding(2) var<storage, read_write> states: array<{{F}}>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  // The bound size encodes the batch: 3*K u32 per sequence.
  let total = arrayLength(&markers) / 3u;
  for (var i = lid.x; i < total * D; i += 64u) {
    let g = i / D;
    let b = g / K;
    let d = i - g * D;
    states[i] = x[(b * L + markers[b * 3u * K + (g - b * K)]) * D + d];
  }
}
