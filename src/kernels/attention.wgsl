{{ENABLE}}// DeBERTa-v2 relative attention, one workgroup per (head, query row).
// scores[i,j] = (q_i.k_j + c2p[i,j] + p2c[i,j]) / SCALE over key positions,
// masked softmax (masked pairs -> -1e4, fully masked rows give a uniform
// distribution like the fp32 reference, never NaN), then . v.
//   c2p[i,j] = q_i . pos_key[idx[i,j]]   (content -> position)
//   p2c[i,j] = k_j . pos_query[idx[i,j]] (position -> content)
// pos_key/pos_query are stored [2*SPAN, H*D] row-major (m-major).
// The q row is hoisted into registers once (it would otherwise be re-read
// for every key); fully masked query rows write zeros and skip the loop.
// Batch (K16): B sequences are packed as B*L global rows; row r = b*L + i
// belongs to sequence b, whose keys/values live at global rows b*L + j
// while relative positions and the sliding table stay local (i, j).

override L: u32 = 128u;
override H: u32 = 6u;
override D: u32 = 64u;
override SCALE: f32 = 13.856406; // sqrt(64 * 3)

@group(0) @binding(0) var<storage, read> qkv: array<{{F}}>;
@group(0) @binding(1) var<storage, read> posKey: array<{{F}}>;
@group(0) @binding(2) var<storage, read> posQuery: array<{{F}}>;
@group(0) @binding(3) var<storage, read> relidx: array<u32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> ctx: array<{{F}}>;

// One score per bucket position; the array follows the L override, so a
// L1024 pipeline takes 4 KiB (well under the 16 KiB minimum limit).
var<workgroup> scores: array<f32, L>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h = wid.x;
  let i = wid.y;
  let b = i / L;
  let il = i - b * L;
  let kbase = b * L;
  let hd = H * D;
  let qb = i * 3u * hd + h * D;
  var qreg: array<f32, 64>;
  for (var d = 0u; d < D; d += 1u) {
    qreg[d] = f32(qkv[qb + d]);
  }
  // Masked query rows contribute nothing downstream (their ctx feeds only
  // their own row), so they can be zeroed and skipped.
  if (mask[i] <= 0.5) {
    for (var d = lid.x; d < D; d += 64u) {
      ctx[i * hd + h * D + d] = {{F}}(0.0);
    }
    return;
  }
  for (var j = lid.x; j < L; j += 64u) {
    // Masked keys land at -1e4 regardless of the dot products; skip them.
    if (mask[kbase + j] <= 0.5) {
      scores[j] = -1e4;
      continue;
    }
    var s = 0.0;
    let p = relidx[il * L + j] * hd;
    let kb = (kbase + j) * 3u * hd + (H + h) * D;
    let pk = p + h * D;
    let pq = p + h * D;
    for (var d = 0u; d < D; d += 4u) {
      let q0 = qreg[d];
      let q1 = qreg[d + 1u];
      let q2 = qreg[d + 2u];
      let q3 = qreg[d + 3u];
      let k0 = f32(qkv[kb + d]);
      let k1 = f32(qkv[kb + d + 1u]);
      let k2 = f32(qkv[kb + d + 2u]);
      let k3 = f32(qkv[kb + d + 3u]);
      s += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
      s += q0 * f32(posKey[pk + d]) + q1 * f32(posKey[pk + d + 1u])
        + q2 * f32(posKey[pk + d + 2u]) + q3 * f32(posKey[pk + d + 3u]);
      s += k0 * f32(posQuery[pq + d]) + k1 * f32(posQuery[pq + d + 1u])
        + k2 * f32(posQuery[pq + d + 2u]) + k3 * f32(posQuery[pq + d + 3u]);
    }
    scores[j] = s / SCALE;
  }
  workgroupBarrier();
  var m = -1e30;
  for (var j = lid.x; j < L; j += 64u) { m = max(m, scores[j]); }
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
    let e = exp(scores[j] - mx);
    scores[j] = e;
    sum += e;
  }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var o = 32u; o > 0u; o >>= 1u) {
    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }
    workgroupBarrier();
  }
  let invTotal = 1.0 / red[0];
  workgroupBarrier();
  for (var d = lid.x; d < D; d += 64u) {
    let vb = (2u * H + h) * D + d;
    // The softmax normalization folds into the epilogue (ctx = acc / total):
    // exp() underflows to exactly 0 for masked keys, so the sv != 0 branch
    // still keeps stale rows out, and four independent accumulators shorten
    // the serial FMA dependency.
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    for (var j = 0u; j + 3u < L; j += 4u) {
      let s0 = scores[j];
      let s1 = scores[j + 1u];
      let s2 = scores[j + 2u];
      let s3 = scores[j + 3u];
      if (s0 != 0.0) { acc0 += s0 * f32(qkv[(kbase + j + 0u) * 3u * hd + vb]); }
      if (s1 != 0.0) { acc1 += s1 * f32(qkv[(kbase + j + 1u) * 3u * hd + vb]); }
      if (s2 != 0.0) { acc2 += s2 * f32(qkv[(kbase + j + 2u) * 3u * hd + vb]); }
      if (s3 != 0.0) { acc3 += s3 * f32(qkv[(kbase + j + 3u) * 3u * hd + vb]); }
    }
    var acc = (acc0 + acc1 + acc2 + acc3) * invTotal;
    ctx[i * hd + h * D + d] = {{F}}(acc);
  }
}
