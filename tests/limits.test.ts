// Static audit: every kernel and dispatch must fit the WebGPU minimum
// limits for every supported bucket (docs/PLAN.md K5). These are the same
// values src/device.ts requests in limits 'minimum' mode.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';

// Model variants: [hidden, heads, intermediate, headHidden].
// small: 384/6/1536/768, base+multi: 768/12/3072/1536.
const VARIANTS = [
  { HIDDEN: 384, HEADS: 6, INTERMEDIATE: 1536, HEAD_HIDDEN: 768 },
  { HIDDEN: 768, HEADS: 12, INTERMEDIATE: 3072, HEAD_HIDDEN: 1536 },
];
// L1280 covers wide schemas (banking77 with 72 labels ~ up to 1.1k tokens).
const BUCKETS = [64, 128, 256, 512, 1024, 1280];
const LAYERS = 12;
const K_MAX = 80;

const LIMITS = {
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
  maxComputeWorkgroupStorageSize: 16384,
  maxStorageBuffersPerShaderStage: 8,
  maxStorageBufferBindingSize: 134217728,
};

const sources = readdirSync('src/kernels')
  .filter((f) => f.endsWith('.wgsl'))
  .map((f) => ({ name: f, code: readFileSync(`src/kernels/${f}`, 'utf8') }));

test('workgroup sizes stay within 256 invocations per axis and total', () => {
  for (const { name, code } of sources) {
    for (const m of code.matchAll(/@workgroup_size\(([^)]+)\)/g)) {
      const dims = m[1].split(',').map((s) => Number(s.trim()));
      const [x = 1, y = 1, z = 1] = dims;
      assert.ok(x <= LIMITS.maxComputeWorkgroupSizeX, `${name}: x=${x}`);
      assert.ok(y <= LIMITS.maxComputeWorkgroupSizeY, `${name}: y=${y}`);
      assert.ok(z <= LIMITS.maxComputeWorkgroupSizeZ, `${name}: z=${z}`);
      assert.ok(
        x * y * z <= LIMITS.maxComputeInvocationsPerWorkgroup,
        `${name}: ${x}x${y}x${z} = ${x * y * z} invocations`);
    }
    assert.match(code, /@workgroup_size\(/, `${name}: no workgroup_size`);
  }
});

test('workgroup storage stays within 16 KiB on every bucket', () => {
  // Per-kernel workgroup bytes as a function of bucket length L.
  const perKernel = {
    'matmul.wgsl': () => 2 * 256 * 4,               // ta + tw tiles
    'attention.wgsl': (L: number) => L * 4 + 64 * 4, // scores[L] + red[64]
    'mbattention.wgsl': (L: number) => L * 4 + 64 * 4, // scores[L] + red[64]
    'layernorm.wgsl': () => 64 * 4,                  // red[64]
    'gather.wgsl': () => 0,
    'masklogits.wgsl': () => 0,
    'rope.wgsl': () => 0,
    'geglu.wgsl': () => 0,
    'add.wgsl': () => 0,
  };
  for (const [name, bytes] of Object.entries(perKernel)) {
    for (const L of BUCKETS) {
      const b = bytes(L);
      assert.ok(
        b <= LIMITS.maxComputeWorkgroupStorageSize,
        `${name} at L${L}: ${b} bytes`);
    }
  }
  // Cross-check: the counted workgroup vars actually exist in the sources.
  for (const { name, code } of sources) {
    const vars = [...code.matchAll(/var<workgroup>\s+(\w+):\s*array<f32,\s*(\w+)>/g)];
    for (const [, ident] of vars) {
      assert.ok(
        perKernel[name as keyof typeof perKernel] !== undefined,
        `${name}: unaudited workgroup var ${ident}`);
    }
  }
});

test('storage buffers per shader stage stay within 8', () => {
  for (const { name, code } of sources) {
    const storages = code.match(/var<storage/g) ?? [];
    assert.ok(
      storages.length <= LIMITS.maxStorageBuffersPerShaderStage,
      `${name}: ${storages.length} storage bindings`);
  }
});

test('dispatch dimensions stay within 65535 workgroups on every bucket', () => {
  for (const v of VARIANTS) {
    for (const L of BUCKETS) {
      const R16 = Math.ceil(L / 16);
      const dispatches: [string, number, number][] = [
        ['ln-emb', L, 1],
        ['mm-qkv', Math.ceil(3 * v.HIDDEN / 16), R16],
        ['attn', v.HEADS, L],
        ['mm-attn-out', Math.ceil(v.HIDDEN / 16), R16],
        ['ln-layer', L, 1],
        ['mm-ffn1', Math.ceil(v.INTERMEDIATE / 16), R16],
        ['mm-ffn2', Math.ceil(v.HIDDEN / 16), R16],
        ['gather', 1, 1],
        ['mm-fc1', Math.ceil(v.HEAD_HIDDEN / 16), 1],
        ['mm-fc2', 1, 1],
        ['masklogits', 1, 1],
      ];
      for (const [name, dx, dy] of dispatches) {
        assert.ok(
          dx <= LIMITS.maxComputeWorkgroupsPerDimension,
          `${name} at L${L} h${v.HIDDEN}: dx=${dx}`);
        assert.ok(
          dy <= LIMITS.maxComputeWorkgroupsPerDimension,
          `${name} at L${L} h${v.HIDDEN}: dy=${dy}`);
      }
    }
  }
});

test('no shader-bound buffer exceeds the 128 MiB binding limit', () => {
  const MiB = 1024 * 1024;
  for (const v of VARIANTS) {
    for (const L of BUCKETS) {
      const bound: [string, number][] = [
        ['emb/x/tmp', L * v.HIDDEN * 4],          // f32 worst case
        ['qkv', L * 3 * v.HIDDEN * 4],
        ['ctx/attnOut/ffnOut', L * v.HIDDEN * 4],
        ['mid', L * v.INTERMEDIATE * 4],
        ['relidx', L * L * 4],
        ['mask', L * 4],
        ['states', K_MAX * v.HIDDEN * 4],
        ['h1', K_MAX * v.HEAD_HIDDEN * 4],
        ['weight ffn_in', v.INTERMEDIATE * v.HIDDEN * 4],
        ['weight qkv', 3 * v.HIDDEN * v.HIDDEN * 4],
        ['weight fc1', v.HEAD_HIDDEN * v.HIDDEN * 4],
      ];
      for (const [name, bytes] of bound) {
        assert.ok(
          bytes <= LIMITS.maxStorageBufferBindingSize,
          `${name} at L${L} h${v.HIDDEN}: ${(bytes / MiB).toFixed(1)} MiB`);
      }
      // The capture buffer is not a shader binding but must fit maxBufferSize.
      assert.ok(
        (LAYERS + 2) * L * v.HIDDEN * 4 <= 268435456,
        `capBuf at L${L} h${v.HIDDEN} exceeds maxBufferSize`);
    }
  }
});
