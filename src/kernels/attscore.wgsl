{{ENABLE}}// Attention scores as a matmul (K27): S[i][h][j] = SCALE * (q_i . k_j) in f32 for every
// head h, query row i and key j of the same sequence. Register-blocked like mmtile.wgsl:
// 16x8 threads, 4x4 per thread, tile 32 query rows x 64 keys, k (the head dimension) in
// steps of 32 through workgroup memory. Workgroup x = h * ceil(L / 64) + key tile, y = query
// block of 32 rows. Layout of qkv: row i holds [q | k | v] of H * D each, B sequences packed
// as B * L rows. Needs D % 32 == 0 and L % 32 == 0. Masking happens in attsoftmax.wgsl; a key
// tile with no valid key writes nothing (its scores are never read).

override L: u32 = 128u;
override H: u32 = 12u;
override D: u32 = 64u;
override SCALE: f32 = 0.125;
override ROWS: u32 = 128u;   // B * L

@group(0) @binding(0) var<storage, read> qkv: array<vec4<{{F}}>>;
@group(0) @binding(1) var<storage, read> kinfo: array<u32>; // first, last valid key per sequence
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;

var<workgroup> sa: array<vec4<{{F}}>, 256>; // [row][d/4], 32 rows x 8
var<workgroup> sw: array<vec4<{{F}}>, 512>; // [d][key/4], 32 x 16
var<workgroup> tileLive: u32;

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let nkt = (L + 63u) / 64u;
  let h = wid.x / nkt;
  let j0 = (wid.x % nkt) * 64u;
  let tid = lid.y * 16u + lid.x;
  let row0 = wid.y * 32u;
  let kbase = (row0 / L) * L;
  let tileRow = lid.y * 4u;
  let D4 = D / 4u;
  let row4 = 3u * H * D4;
  // A key tile outside the valid keys of the sequence is never read (attsoftmax skips masked
  // keys): skip it.
  if (tid == 0u) {
    let b = row0 / L;
    let first = kinfo[2u * b];
    let last = kinfo[2u * b + 1u];
    tileLive = select(0u, 1u, first < L && j0 <= last && j0 + 63u >= first);
  }
  if (workgroupUniformLoad(&tileLive) == 0u) { return; }
  var acc: array<vec4<f32>, 4>;
  for (var t4 = 0u; t4 < D4; t4 += 8u) {
    for (var run = tid; run < 256u; run += 128u) {
      let r = row0 + run / 8u;
      sa[run] = select(vec4<{{F}}>(0.0), qkv[r * row4 + h * D4 + t4 + run % 8u], r < ROWS);
    }
    // keys in groups of 4 (transposed in registers), 16 groups x 8 d chunks
    let ng = tid / 8u;
    let kv = tid % 8u;
    var m: array<vec4<{{F}}>, 4>;
    for (var q = 0u; q < 4u; q += 1u) {
      let j = j0 + ng * 4u + q;
      m[q] = select(vec4<{{F}}>(0.0), qkv[(kbase + j) * row4 + (H + h) * D4 + t4 + kv], j < L);
    }
    for (var e = 0u; e < 4u; e += 1u) {
      sw[(kv * 4u + e) * 16u + ng] = vec4<{{F}}>(m[0][e], m[1][e], m[2][e], m[3][e]);
    }
    workgroupBarrier();
    for (var k4 = 0u; k4 < 8u; k4 += 1u) {
      let b0 = vec4<f32>(sw[(k4 * 4u) * 16u + lid.x]);
      let b1 = vec4<f32>(sw[(k4 * 4u + 1u) * 16u + lid.x]);
      let b2 = vec4<f32>(sw[(k4 * 4u + 2u) * 16u + lid.x]);
      let b3 = vec4<f32>(sw[(k4 * 4u + 3u) * 16u + lid.x]);
      for (var i = 0u; i < 4u; i += 1u) {
        let av = vec4<f32>(sa[(tileRow + i) * 8u + k4]);
        acc[i] = b0 * av.x + acc[i];
        acc[i] = b1 * av.y + acc[i];
        acc[i] = b2 * av.z + acc[i];
        acc[i] = b3 * av.w + acc[i];
      }
    }
    workgroupBarrier();
  }
  let j = j0 + lid.x * 4u;
  for (var i = 0u; i < 4u; i += 1u) {
    let row = row0 + tileRow + i;
    if (row >= ROWS) { continue; }
    for (var c = 0u; c < 4u; c += 1u) {
      if (j + c < L) {
        scores[(row * H + h) * L + j + c] = acc[i][c] * SCALE;
      }
    }
  }
}
