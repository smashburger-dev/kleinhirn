{{ENABLE}}// Standard multi-head attention, flash form (K27): one workgroup per (head, block of 32
// query rows), one thread per query row. Keys and values stream through workgroup memory in
// blocks of 32 (vec4 of the storage type, 2 x 32 x D values); every thread keeps its q row
// and its output row in registers and runs an online softmax in f32. Masked keys and keys
// outside the sliding window (WINDOW > 0, |i - j| <= WINDOW) are skipped, so stale padding
// rows never enter a sum; masked query rows write zeros. Layout as mbattention.wgsl: row i
// holds [q | k | v] of H * D each, B sequences packed as B * L rows. Needs D % 4 == 0,
// D <= 64 and L % 32 == 0 (a block never straddles two sequences).

override L: u32 = 128u;
override H: u32 = 6u;
override D: u32 = 64u;
override SCALE: f32 = 0.125; // 64^-0.5
override WINDOW: u32 = 0u;   // 0 = global attention
override ROWS: u32 = 128u;   // B * L, rows of qkv

@group(0) @binding(0) var<storage, read> qkv: array<vec4<{{F}}>>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> ctx: array<vec4<{{F}}>>;

var<workgroup> sk: array<vec4<{{F}}>, 512>; // [key][d/4], 32 keys x 16
var<workgroup> sv: array<vec4<{{F}}>, 512>;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h = wid.x;
  let i = wid.y * 32u + lid.x;
  let b = (wid.y * 32u) / L;   // the whole block lies in sequence b
  let il = i - b * L;
  let kbase = b * L;
  let D4 = D / 4u;
  let hd4 = H * D4;            // vec4 per q, k or v part of a row
  let row4 = 3u * hd4;
  let live = i < ROWS && mask[min(i, ROWS - 1u)] > 0.5;
  var q: array<vec4<f32>, 16>;
  var o: array<vec4<f32>, 16>;
  if (live) {
    for (var d = 0u; d < D4; d += 1u) {
      q[d] = vec4<f32>(qkv[i * row4 + h * D4 + d]) * SCALE;
    }
  }
  var m = -1e30;
  var l = 0.0;
  for (var j0 = 0u; j0 < L; j0 += 32u) {
    // 32 keys x D4 vec4 per operand, 32 threads.
    for (var e = lid.x; e < 32u * D4; e += 32u) {
      let kj = kbase + j0 + e / D4;
      let d = e % D4;
      sk[e] = qkv[kj * row4 + (H + h) * D4 + d];
      sv[e] = qkv[kj * row4 + (2u * H + h) * D4 + d];
    }
    workgroupBarrier();
    if (live) {
      for (var jj = 0u; jj < 32u; jj += 1u) {
        let j = j0 + jj;
        var skip = mask[kbase + j] <= 0.5;
        if (WINDOW > 0u) {
          let dist = select(il - j, j - il, j > il);
          skip = skip || dist > WINDOW;
        }
        if (skip) { continue; }
        var s = 0.0;
        for (var d = 0u; d < D4; d += 1u) {
          s += dot(q[d], vec4<f32>(sk[jj * D4 + d]));
        }
        if (s > m) {
          let c = exp(m - s);
          l *= c;
          for (var d = 0u; d < D4; d += 1u) { o[d] *= c; }
          m = s;
        }
        let p = exp(s - m);
        l += p;
        for (var d = 0u; d < D4; d += 1u) {
          o[d] += p * vec4<f32>(sv[jj * D4 + d]);
        }
      }
    }
    workgroupBarrier();
  }
  if (i < ROWS) {
    let inv = select(0.0, 1.0 / l, live);
    for (var d = 0u; d < D4; d += 1u) {
      ctx[i * hd4 + h * D4 + d] = vec4<{{F}}>(o[d] * inv);
    }
  }
}
