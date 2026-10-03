// Attention softmax over the rows of attscore.wgsl (K27), in place and in f32: one workgroup
// per (head, query row). Keys with mask 0 and keys outside the sliding window (WINDOW > 0,
// |i - j| <= WINDOW) get probability exactly 0; a masked query row gets 0 everywhere (its
// context row is zero, as in mbattention.wgsl). Every unmasked query has at least itself as
// a key, so the sum is never 0 there.

override L: u32 = 128u;
override H: u32 = 12u;
override WINDOW: u32 = 0u; // 0 = global attention

@group(0) @binding(0) var<storage, read> mask: array<f32>;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;

var<workgroup> red: array<f32, 64>;

fn keep(il: u32, j: u32, kbase: u32) -> bool {
  var k = mask[kbase + j] > 0.5;
  if (WINDOW > 0u) {
    let dist = select(il - j, j - il, j > il);
    k = k && dist <= WINDOW;
  }
  return k;
}

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
  var m = -1e30;
  for (var j = lid.x; j < L; j += 64u) {
    if (keep(il, j, kbase)) { m = max(m, scores[base + j]); }
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
    if (keep(il, j, kbase)) { e = exp(scores[base + j] - mx); }
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
