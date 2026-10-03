{{ENABLE}}// Embedding sum and LayerNorm, one workgroup per row, f32 accumulation.
// out[row, :] = LN(word[row, :] + pos[(row % L) + OFFSET, :] + typ[tt[row], :])
//   * (MASKMUL == 1 ? mask[row] : 1)
// The three-table sum is formed in f32; the result is rounded once on write.
// L is the bucket length: in a batch, row r = b * L + i reads position i.
// Rows past the position table (a bucket longer than MAXPOS - OFFSET) are padding and masked: the
// position index is clamped to the last row, valid rows (i < MAXPOS - OFFSET) never reach the clamp.
// Families without a type table bind a zero row and zero ids.

override N: u32 = 384u;
override L: u32 = 128u;
override OFFSET: u32 = 0u;
override MAXPOS: u32 = 0xffffffffu;
override EPS: f32 = 1e-12;
override MASKMUL: u32 = 0u;

@group(0) @binding(0) var<storage, read> word: array<{{F}}>;
@group(0) @binding(1) var<storage, read> pos: array<{{F}}>;
@group(0) @binding(2) var<storage, read> typ: array<{{F}}>;
@group(0) @binding(3) var<storage, read> tt: array<u32>;
@group(0) @binding(4) var<storage, read> weight: array<{{F}}>;
@group(0) @binding(5) var<storage, read> bias: array<{{F}}>;
@group(0) @binding(6) var<storage, read> mask: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<{{F}}>;

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let base = row * N;
  let pbase = min((row % L) + OFFSET, MAXPOS - 1u) * N;
  let tbase = tt[row] * N;
  var s = 0.0;
  var sq = 0.0;
  for (var i = lid.x; i < N; i += 64u) {
    let v = f32(word[base + i]) + f32(pos[pbase + i]) + f32(typ[tbase + i]);
    s += v;
    sq += v * v;
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) {
    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }
    workgroupBarrier();
  }
  let mean = red[0] / f32(N);
  workgroupBarrier();
  red[lid.x] = sq;
  workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) {
    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }
    workgroupBarrier();
  }
  let variance = red[0] / f32(N) - mean * mean;
  let inv = 1.0 / sqrt(variance + EPS);
  for (var i = lid.x; i < N; i += 64u) {
    let v = f32(word[base + i]) + f32(pos[pbase + i]) + f32(typ[tbase + i]);
    var y = (v - mean) * inv * f32(weight[i]) + f32(bias[i]);
    if (MASKMUL == 1u) { y *= mask[row]; }
    out[base + i] = {{F}}(y);
  }
}
