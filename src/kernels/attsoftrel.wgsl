// DeBERTa attention softmax (K27), in place and in f32: one workgroup per (head, query row).
// The scores of attscore.wgsl hold (q_i . k_j) / SCALE; this kernel adds the relative terms
// (c2p[i][h][m] + p2c[j][h][m]) / SCALE with m = relidx[i][j] - MOFF (attrel.wgsl), then
// runs the masked softmax: keys with mask 0 get probability exactly 0, a masked query row
// gets 0 everywhere. Same score as attention.wgsl: (q.k + q.pos_key[m] + k.pos_query[m]) / SCALE.

override L: u32 = 128u;
override H: u32 = 12u;
override NM: u32 = 256u;
override MOFF: u32 = 0u;
override INVSCALE: f32 = 0.0721688; // 1 / sqrt(64 * 3)

@group(0) @binding(0) var<storage, read> mask: array<f32>;
@group(0) @binding(1) var<storage, read> relidx: array<u32>;
@group(0) @binding(2) var<storage, read> c2p: array<f32>;
@group(0) @binding(3) var<storage, read> p2c: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h = wid.x;
  let i = wid.y;
  let kbase = (i / L) * L;
  let il = i - kbase;
  let base = (i * H + h) * L;
  let cbase = (i * H + h) * NM;
  var m = -3.40282346638528859812e38f; // lowest finite f32: below every real score (review R21)
  for (var j = lid.x; j < L; j += 64u) {
    if (mask[kbase + j] > 0.5) {
      let p = relidx[il * L + j] - MOFF;
      let s = scores[base + j] + (c2p[cbase + p] + p2c[((kbase + j) * H + h) * NM + p]) * INVSCALE;
      scores[base + j] = s;
      m = max(m, s);
    }
  }
  red[lid.x] = m;
  workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) {
    if (lid.x < o) { red[lid.x] = max(red[lid.x], red[lid.x + o]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lid.x; j < L; j += 64u) {
    var e = 0.0;
    if (mask[kbase + j] > 0.5) { e = exp(scores[base + j] - mx); }
    scores[base + j] = e;
    sum += e;
  }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) {
    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }
    workgroupBarrier();
  }
  let inv = select(0.0, 1.0 / red[0], mask[i] > 0.5 && red[0] > 0.0);
  for (var j = lid.x; j < L; j += 64u) {
    scores[base + j] *= inv;
  }
}
