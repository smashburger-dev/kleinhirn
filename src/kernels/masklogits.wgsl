{{ENABLE}}// logits[k] = raw[k] / TEMP where markerMask[k] > 0.5 else -1e4.
// Marker payloads are packed u32: [0..K) marker_indices, [K..2K)
// marker_mask as f32 bits, [2K..3K) marker_groups (resolved on the CPU).
// Batch (K16): packed holds B blocks of 3*K; output k maps to sequence
// b = k / K and mask element b*3K + K + (k mod K).

override K: u32 = 16u;
override TEMP: f32 = 1.0;

@group(0) @binding(0) var<storage, read> raw: array<{{F}}>;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let total = arrayLength(&packed) / 3u;
  for (var k = lid.x; k < total; k += 64u) {
    let b = k / K;
    let m = bitcast<f32>(packed[b * 3u * K + K + (k - b * K)]);
    logits[k] = select(-1e4, f32(raw[k]) / TEMP, m > 0.5);
  }
}
