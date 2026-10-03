{{ENABLE}}// Masked pooling per sequence, f32 accumulation. One workgroup per
// (64 columns, sequence); each thread owns one column and loops over the L
// rows of its sequence. Rows of sequence b are b*L .. b*L + L - 1.
// MODE 0: out[b, d] = sum_i mask * x / max(sum_i mask, 1e-9)   (mean)
// MODE 1: out[b, d] = max over i with mask > 0.5 of x           (max)
// The 1e-9 floor is the clamp of sentence-transformers. MODE 1 starts at the
// first valid row (a found flag, no sentinel), so every finite value is a
// valid maximum; a sequence without a valid row gives 0.

override L: u32 = 128u;
override N: u32 = 384u;
override MODE: u32 = 0u;

@group(0) @binding(0) var<storage, read> x: array<{{F}}>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<{{F}}>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let d = wid.x * 64u + lid.x;
  let b = wid.y;
  if (d >= N) { return; }
  let rbase = b * L;
  if (MODE == 0u) {
    var acc = 0.0;
    var cnt = 0.0;
    for (var i = 0u; i < L; i += 1u) {
      let m = mask[rbase + i];
      // Rows past seqLen hold stale values; skip them instead of 0 * stale.
      if (m != 0.0) {
        acc += m * f32(x[(rbase + i) * N + d]);
        cnt += m;
      }
    }
    out[b * N + d] = {{F}}(acc / max(cnt, 1e-9));
  } else {
    var best = 0.0;
    var found = false;
    for (var i = 0u; i < L; i += 1u) {
      if (mask[rbase + i] > 0.5) {
        let v = f32(x[(rbase + i) * N + d]);
        best = select(v, max(best, v), found);
        found = true;
      }
    }
    out[b * N + d] = {{F}}(best);
  }
}
