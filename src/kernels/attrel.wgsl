{{ENABLE}}// DeBERTa relative terms as a matmul (K27): OUT[r][h][m] = x_r . T[MOFF + m] in f32, x the
// q part (PART 0, content -> position: c2p with T = pos_key) or the k part (PART 1, position
// -> content: p2c with T = pos_query) of qkv row r, for the NM position rows the bucket can
// reach (MOFF .. MOFF + NM - 1). T is [2 * SPAN, H * D] row-major. Register-blocked like
// attscore.wgsl: 16x8 threads, 4x4 per thread, tile 32 rows x 64 positions, the head
// dimension in steps of 32. Workgroup x = h * ceil(NM / 64) + position tile, y = block of 32
// rows. Needs D % 32 == 0. Position tiles no valid pair of the sequence reaches write nothing.

override H: u32 = 12u;
override D: u32 = 64u;
override ROWS: u32 = 128u;   // B * L
override NM: u32 = 256u;
override MOFF: u32 = 0u;
override PART: u32 = 0u;
override L: u32 = 128u;      // bucket length (relidx is L x L)

@group(0) @binding(0) var<storage, read> qkv: array<vec4<{{F}}>>;
@group(0) @binding(1) var<storage, read> table: array<vec4<{{F}}>>;
@group(0) @binding(2) var<storage, read> kinfo: array<u32>; // first, last valid key per sequence
@group(0) @binding(3) var<storage, read> relidx: array<u32>;
@group(0) @binding(4) var<storage, read_write> rel: array<f32>;

var<workgroup> sa: array<vec4<{{F}}>, 256>; // [row][d/4], 32 rows x 8
var<workgroup> sw: array<vec4<{{F}}>, 512>; // [d][m/4], 32 x 16
var<workgroup> tileLive: u32;

@compute @workgroup_size(16, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let nmt = (NM + 63u) / 64u;
  let h = wid.x / nmt;
  let m0 = (wid.x % nmt) * 64u;
  let tid = lid.y * 16u + lid.x;
  let row0 = wid.y * 32u;
  let tileRow = lid.y * 4u;
  let D4 = D / 4u;
  let hd4 = H * D4;
  let row4 = 3u * hd4;
  // Valid keys of this sequence span first .. last; their pairs reach the positions from
  // relidx[first][last] to relidx[last][first] (the bucket is monotone in i - j). A position
  // tile outside that range is never read (attsoftrel reads valid pairs only): skip it.
  if (tid == 0u) {
    let b = row0 / L;
    let first = kinfo[2u * b];
    let last = kinfo[2u * b + 1u];
    var live = 0u;
    if (first < L) {
      let lo = relidx[first * L + last] - MOFF;
      let hi = relidx[last * L + first] - MOFF;
      live = select(0u, 1u, m0 <= hi && m0 + 63u >= lo);
    }
    tileLive = live;
  }
  if (workgroupUniformLoad(&tileLive) == 0u) { return; }
  var acc: array<vec4<f32>, 4>;
  for (var t4 = 0u; t4 < D4; t4 += 8u) {
    for (var run = tid; run < 256u; run += 128u) {
      let r = row0 + run / 8u;
      sa[run] = select(vec4<{{F}}>(0.0), qkv[r * row4 + (PART * H + h) * D4 + t4 + run % 8u], r < ROWS);
    }
    let ng = tid / 8u;
    let kv = tid % 8u;
    var m: array<vec4<{{F}}>, 4>;
    for (var q = 0u; q < 4u; q += 1u) {
      let mm = m0 + ng * 4u + q;
      m[q] = select(vec4<{{F}}>(0.0), table[(MOFF + mm) * hd4 + h * D4 + t4 + kv], mm < NM);
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
  let mc = m0 + lid.x * 4u;
  for (var i = 0u; i < 4u; i += 1u) {
    let row = row0 + tileRow + i;
    if (row >= ROWS) { continue; }
    for (var c = 0u; c < 4u; c += 1u) {
      if (mc + c < NM) {
        rel[(row * H + h) * NM + mc + c] = acc[i][c];
      }
    }
  }
}
