{{ENABLE}}// im2col for a 1-D convolution over packed sequences, one workgroup per row.
// Row r = b * L + i of out holds the KS rows emb[b * L + i + t - PAD], t = 0..KS-1, side by side
// (block t at columns t * N); a source position outside [0, L) of the own sequence reads as zero, and so
// does a source row with mask 0: the executor dispatches only the rows below seqLen, so a padding row
// of emb can hold stale values of an earlier call instead of the zeros of the embedding.
// emb is [rows, N], out is [rows, KS * N]. Pure copy, no arithmetic.

override N: u32 = 768u;
override L: u32 = 128u;
override KS: u32 = 3u;

@group(0) @binding(0) var<storage, read> emb: array<{{F}}>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<{{F}}>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let seq = row / L;
  let i = i32(row % L);
  let pad = i32((KS - 1u) / 2u);
  for (var t = 0u; t < KS; t += 1u) {
    let j = i + i32(t) - pad;
    let srcRow = seq * L + u32(max(j, 0));
    let ok = j >= 0 && j < i32(L) && mask[srcRow] != 0.0;
    let src = srcRow * N;
    for (var c = lid.x; c < N; c += 64u) {
      var v = {{F}}(0.0);
      if (ok) { v = emb[src + c]; }
      out[row * KS * N + t * N + c] = v;
    }
  }
}
