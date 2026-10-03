{{ENABLE}}// Attention context as a matmul (K27): ctx[i][h * D + d] = sum_j P[i][h][j] * v_j[h * D + d],
// P the f32 probabilities of attsoftmax.wgsl. Register-blocked: 8x8 threads, 4x4 per thread,
// tile 32 query rows x 32 head dims, keys in steps of 32 through workgroup memory. Value rows
// of masked keys load as zero, so stale padding rows never enter the sum (their P is 0 too).
// Workgroup x = h * (D / 32) + dim tile, y = query block of 32 rows. Needs D % 32 == 0 and
// L % 32 == 0. Every output sums j in order in f32.

override L: u32 = 128u;
override H: u32 = 12u;
override D: u32 = 64u;
override ROWS: u32 = 128u;   // B * L

@group(0) @binding(0) var<storage, read> probs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> qkv: array<vec4<{{F}}>>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read> kinfo: array<u32>; // first, last valid key per sequence
@group(0) @binding(4) var<storage, read_write> ctx: array<{{F}}>;

var<workgroup> sa: array<vec4<f32>, 256>;   // [row][key/4], 32 x 8
var<workgroup> sb: array<vec4<{{F}}>, 256>; // [key][d/4], 32 x 8
var<workgroup> keyFirst: u32;
var<workgroup> keyLast: u32;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let nd = D / 32u;
  let h = wid.x / nd;
  let dt = wid.x % nd;
  let tid = lid.y * 8u + lid.x;
  let row0 = wid.y * 32u;
  let kbase = (row0 / L) * L;
  let tileRow = lid.y * 4u;
  let L4 = L / 4u;
  let D4 = D / 4u;
  let row4 = 3u * H * D4;
  var acc: array<vec4<f32>, 4>;
  // 32 keys outside the valid keys of the sequence add exactly zero (P is 0 there): skip them.
  if (tid == 0u) {
    keyFirst = kinfo[2u * (row0 / L)];
  }
  let first = workgroupUniformLoad(&keyFirst);
  if (tid == 0u) {
    keyLast = kinfo[2u * (row0 / L) + 1u];
  }
  let last = workgroupUniformLoad(&keyLast);
  for (var t4 = 0u; t4 < L4; t4 += 8u) {
    if (first >= L || t4 * 4u > last || t4 * 4u + 31u < first) { continue; }
    for (var run = tid; run < 256u; run += 64u) {
      let r = row0 + run / 8u;
      sa[run] = select(vec4<f32>(0.0), probs[(r * H + h) * L4 + t4 + run % 8u], r < ROWS);
      let key = kbase + t4 * 4u + run / 8u;
      sb[run] = select(vec4<{{F}}>(0.0), qkv[key * row4 + (2u * H + h) * D4 + dt * 8u + run % 8u],
        mask[key] > 0.5);
    }
    workgroupBarrier();
    for (var k4 = 0u; k4 < 8u; k4 += 1u) {
      let b0 = vec4<f32>(sb[(k4 * 4u) * 8u + lid.x]);
      let b1 = vec4<f32>(sb[(k4 * 4u + 1u) * 8u + lid.x]);
      let b2 = vec4<f32>(sb[(k4 * 4u + 2u) * 8u + lid.x]);
      let b3 = vec4<f32>(sb[(k4 * 4u + 3u) * 8u + lid.x]);
      for (var i = 0u; i < 4u; i += 1u) {
        let av = sa[(tileRow + i) * 8u + k4];
        acc[i] = b0 * av.x + acc[i];
        acc[i] = b1 * av.y + acc[i];
        acc[i] = b2 * av.z + acc[i];
        acc[i] = b3 * av.w + acc[i];
      }
    }
    workgroupBarrier();
  }
  let col = h * D + dt * 32u + lid.x * 4u;
  for (var i = 0u; i < 4u; i += 1u) {
    let row = row0 + tileRow + i;
    if (row >= ROWS) { continue; }
    for (var c = 0u; c < 4u; c += 1u) {
      ctx[row * H * D + col + c] = {{F}}(acc[i][c]);
    }
  }
}
