// Static audit: every kernel and dispatch must fit the WebGPU minimum
// limits for every supported bucket (docs/PLAN.md K5). These are the same
// values src/device.ts requests in limits 'minimum' mode.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { buildPlan } from '../src/plan/build.ts';
import { MAX_WORKGROUPS, addWorkgroups } from '../src/plan/layers.ts';
import { specFromGlinerManifest, specFromJuliaManifest } from '../src/plan/spec.ts';
import { bindingMismatches, declaredBindings } from './helpers/bindings.ts';
import { GLINER_HEAD_HIDDEN, GLINER_SPECS, JULIA_SPEC } from './helpers/plan-trace.ts';
import { EMBED_CHUNK_ROWS } from '../src/convert/manifest.ts';
import { PILOT_IDS, PILOT_K285_IDS, PILOT_K286_IDS, pilotSpec } from './helpers/pilot.ts';

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
    'attscore.wgsl': () => (256 + 512) * 16 + 4,     // K27, f32 plans, tileLive
    'attsoftmax.wgsl': () => 64 * 4,                 // red[64]
    'attrel.wgsl': () => (256 + 512) * 16 + 4,       // K27, f32 plans, tileLive
    'attsoftrel.wgsl': () => 64 * 4,                 // red[64]
    'attpv.wgsl': () => 2 * 256 * 16 + 8,            // K27, f32 plans, keyFirst, keyLast
    'mbflash.wgsl': () => 2 * 512 * 16,              // sk + sv vec4 tiles, f32 plans (K27)
    'mmtile.wgsl': () => (256 + 512) * 16,          // sa + sw vec4 tiles, f32 plans (K27)
    'mmtile16.wgsl': () => (128 + 256) * 16,        // sa + sw vec4 tiles, f32 plans (K27)
    'mmtile8.wgsl': () => (64 + 256) * 16,          // sa + sw vec4 tiles, f32 plans (K27)
    'attention.wgsl': (L: number) => L * 4 + 64 * 4, // scores[L] + red[64]
    'mbattention.wgsl': (L: number) => L * 4 + 64 * 4, // scores[L] + red[64]
    'layernorm.wgsl': () => 64 * 4,                  // red[64]
    'gather.wgsl': () => 0,
    'masklogits.wgsl': () => 0,
    'rope.wgsl': () => 0,
    'geglu.wgsl': () => 0,
    'add.wgsl': () => 0,
    'embln.wgsl': () => 64 * 4,                      // red[64]
    'pool.wgsl': () => 0,
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

// K28.4: pilot shapes (H 256/384/768, E 128, heads 4/8/12, head width 32/64,
// buckets 128/256/512, B up to 16, 1 to 9 classes) through the new kernels.
const PILOT_H = [256, 384, 768];
const PILOT_HEADS: [number, number][] = [[256, 4], [256, 8], [384, 12], [768, 12]];
const PILOT_BUCKETS = [128, 256, 512];
const PILOT_B = [1, 4, 8, 16];
const ceil = (n: number, d: number): number => Math.ceil(n / d);

test('embln and pool declare at most 8 storage bindings and 64 threads', () => {
  for (const name of ['embln', 'pool'] as const) {
    const code = readFileSync(`src/kernels/${name}.wgsl`, 'utf8');
    assert.match(code, /@workgroup_size\(64\)/, name);
    assert.ok((code.match(/var<storage/g) ?? []).length <= 8, name);
  }
  assert.equal(declaredBindings('embln'), 8);
  assert.equal(declaredBindings('pool'), 3);
});

test('pilot dispatch dimensions stay within 65535', () => {
  for (const B of PILOT_B) {
    for (const L of PILOT_BUCKETS) {
      const rows = B * L;
      for (const H of PILOT_H) {
        const dispatches: [string, number, number][] = [
          ['embln', rows, 1],
          ['mm-qkv', ceil(3 * H, 16), ceil(rows, 16)],
          ['mm-ffn1', ceil(4 * H, 16), ceil(rows, 16)],
          ['attn', 12, rows],
          ['pool', ceil(H, 64), B],
          ['mm-pooler', ceil(H, 16), ceil(B, 16)],
          ['mm-token-head', 1, ceil(rows, 16)],
        ];
        for (const [name, dx, dy] of dispatches) {
          assert.ok(dx <= LIMITS.maxComputeWorkgroupsPerDimension, `${name} B${B} L${L} H${H}: dx=${dx}`);
          assert.ok(dy <= LIMITS.maxComputeWorkgroupsPerDimension, `${name} B${B} L${L} H${H}: dy=${dy}`);
        }
      }
    }
  }
});

test('pilot buffers stay within the 128 MiB binding limit', () => {
  const MiB = 1024 * 1024;
  for (const B of PILOT_B) {
    for (const L of PILOT_BUCKETS) {
      const rows = B * L;
      for (const [H, heads] of PILOT_HEADS) {
        const I = 4 * H;
        const bound: [string, number][] = [
          ['emb', rows * H * 4],
          ['qkv', rows * 3 * H * 4],
          ['mid', rows * I * 4],
          ['x', rows * H * 4],
          ['mask/typeIds', rows * 4],
          ['position table', 512 * H * 4],
          ['token out', rows * 9 * 4],
          ['weight ffn_in', I * H * 4],
          ['weight qkv', 3 * H * H * 4],
        ];
        for (const [name, bytes] of bound) {
          assert.ok(bytes <= LIMITS.maxStorageBufferBindingSize,
            `${name} B${B} L${L} H${H} heads${heads}: ${(bytes / MiB).toFixed(1)} MiB`);
        }
        // mbattention: scores[L] + red[64] with L up to 512 stays far below 16 KiB.
        assert.ok(L * 4 + 64 * 4 <= LIMITS.maxComputeWorkgroupStorageSize);
      }
    }
  }
});

test('every plan operation binds as many buffers as its kernel declares', () => {
  const plans = [];
  for (const shape of ['small', 'base'] as const) {
    const { spec, head } = specFromGlinerManifest(GLINER_SPECS[shape], {
      temperature: 1, hiddenSize: GLINER_HEAD_HIDDEN[shape] });
    for (const f16 of [false, true]) {
      plans.push([`gliner-${shape}`, buildPlan(spec, head, { length: 128, batch: 4, markers: 16, f16 })] as const);
    }
  }
  const julia = specFromJuliaManifest(JULIA_SPEC);
  plans.push(['julia-1', buildPlan(julia.spec, julia.head, { length: 512, batch: 1, markers: 20, f16: true })] as const);
  for (const id of [...PILOT_IDS, ...PILOT_K285_IDS, ...PILOT_K286_IDS]) {
    const { spec, head } = pilotSpec(id);
    for (const [length, batch] of [[128, 1], [512, 4]]) {
      plans.push([id, buildPlan(spec, head, { length, batch, markers: 0, f16: false })] as const);
    }
  }
  for (const [name, plan] of plans) {
    assert.deepEqual(bindingMismatches(plan), [], name);
  }
});

// K28.5: RoBERTa, XLM-R and DistilBERT shapes. H 384 with 12 heads (head width
// 32) is in PILOT_HEADS already; new are the position offset rows, the Dense
// step 768 to 512 and the 250,002-row vocabulary that stays on the CPU.
test('K28.5 position table with offset and Dense dispatch fit the limits', () => {
  for (const B of PILOT_B) {
    for (const L of PILOT_BUCKETS) {
      assert.ok(ceil(512, 16) <= LIMITS.maxComputeWorkgroupsPerDimension);   // Dense out 512
      assert.ok(ceil(B, 16) <= LIMITS.maxComputeWorkgroupsPerDimension);
      assert.ok((L + 2) * 768 * 4 <= LIMITS.maxStorageBufferBindingSize);    // 514 position rows at H 768
    }
  }
});

test('K28.5 vocabulary of 250,002 rows stays on the CPU, no GPU buffer scales with it', () => {
  const xlmr = pilotSpec('MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli');
  assert.equal(xlmr.spec.vocab, 250_002);
  // the word table is joined on the CPU from row chunks (f32 at most 2^31 bytes)
  assert.equal(Math.ceil(250_002 / EMBED_CHUNK_ROWS), 5);
  assert.ok(250_002 * xlmr.spec.embeddingSize * 4 < 2 ** 31);
  for (const f16 of [false, true]) {
    const plan = buildPlan(xlmr.spec, xlmr.head, { length: 512, batch: 16, markers: 0, f16 });
    for (const b of plan.buffers) {
      assert.ok(b.bytes <= LIMITS.maxStorageBufferBindingSize, `${b.id}: ${b.bytes} B`);
      assert.ok(b.bytes < 250_002 * xlmr.spec.embeddingSize, `${b.id} scales with the vocabulary`);
    }
    assert.ok(!plan.segments.some((s) => s.ops.some((o) => o.bind.includes('w:embeddings.word.weight'))));
  }
});

// K28.6: DeBERTa and ModernBERT shapes. New are H 600 with 12 heads (head width 50, a width the
// 4-wide dot product of mbattention does not divide), head width 32 with RoPE (H 384, 12 heads),
// two RoPE tables, the LayerNorm head step and the 256,000-row vocabulary on the CPU.
test('K28.6 plans of every pilot stay within the limits at the largest buckets', () => {
  const MiB = 1024 * 1024;
  for (const id of PILOT_K286_IDS) {
    const { spec, head } = pilotSpec(id);
    for (const [length, batch] of [[512, 16], [1024, 4]]) {
      for (const f16 of [false, true]) {
        const plan = buildPlan(spec, head, { length, batch, markers: 0, f16 });
        assert.deepEqual(bindingMismatches(plan), [], id);
        for (const b of plan.buffers) {
          assert.ok(b.bytes <= LIMITS.maxStorageBufferBindingSize,
            `${id} ${b.id} L${length} B${batch}: ${(b.bytes / MiB).toFixed(1)} MiB`);
        }
        for (const seg of plan.segments) {
          for (const op of seg.ops) {
            for (const d of op.dispatch) {
              const n = d === 'rows' ? length * batch : d === 'rows8' ? Math.ceil(length * batch / 8) : d === 'rows16' ? Math.ceil(length * batch / 16) : d === 'rows32' ? Math.ceil(length * batch / 32) : d;
              assert.ok(n <= LIMITS.maxComputeWorkgroupsPerDimension, `${id} ${seg.prefix}${op.name}: ${n}`);
            }
          }
        }
        // workgroup storage of the attention kernels: scores[L] + red[64]
        assert.ok(length * 4 + 64 * 4 <= LIMITS.maxComputeWorkgroupStorageSize);
        assert.ok(!plan.segments.some((s) => s.ops.some((o) => o.bind.includes('w:embeddings.word.weight'))));
      }
    }
  }
});

test('K28.6 vocabularies stay on the CPU', () => {
  assert.equal(pilotSpec('Horizon-Labs/multilingual-zeroshot-small').spec.vocab, 256_000);
  assert.equal(Math.ceil(256_000 / EMBED_CHUNK_ROWS), 5);
  assert.ok(256_000 * 384 * 4 < 2 ** 31);
  assert.ok(128_100 * 768 * 4 < 2 ** 31);
});

test('K28.6b the residual add dispatch is capped at 65535 and strides over the rest', () => {
  assert.equal(MAX_WORKGROUPS, LIMITS.maxComputeWorkgroupsPerDimension);
  assert.equal(addWorkgroups(512 * 16, 384), 49152);   // below the cap: one workgroup per 64 elements
  assert.equal(addWorkgroups(512 * 16, 512), 65535);   // 65,536 before
  assert.equal(addWorkgroups(1024 * 16, 384), 65535);  // 98,304 before (Julia B16 at L1024)
  const add = readFileSync('src/kernels/add.wgsl', 'utf8');
  assert.match(add, /@builtin\(num_workgroups\)/);
  assert.match(add, /for \(var i = gid\.x; i < TOTAL; i \+= 64u \* nwg\.x\)/);
});

// K28.6b: both dispatch dimensions of every operation of every plan stay within 65535, for
// every pilot, GLiNER small and base and Julia, at buckets up to 1024 and B up to 16 wherever
// the binding limit admits the plan. The residual add used to need one workgroup per 64
// elements and crossed the limit (Julia at L1024 B16: 98,304; H 768 at L512 B16: 65,536).
test('K28.6b every dispatch of every plan stays within 65535 workgroups per dimension', () => {
  const models: [string, ReturnType<typeof pilotSpec> | ReturnType<typeof specFromJuliaManifest>, number][] = [];
  for (const shape of ['small', 'base'] as const) {
    const g = specFromGlinerManifest(GLINER_SPECS[shape], { temperature: 1, hiddenSize: GLINER_HEAD_HIDDEN[shape] });
    models.push([`gliner-${shape}`, g, 16]);
  }
  models.push(['julia-1', specFromJuliaManifest(JULIA_SPEC), 20]);
  for (const id of [...PILOT_IDS, ...PILOT_K285_IDS, ...PILOT_K286_IDS]) models.push([id, pilotSpec(id), 0]);
  assert.equal(models.length, 38);
  const over: string[] = [];
  let built = 0;
  let skipped = 0;
  for (const [id, { spec, head }, markers] of models) {
    for (const length of [128, 256, 512, 1024]) {
      for (const batch of [1, 4, 8, 16]) {
        for (const f16 of [false, true]) {
          const plan = buildPlan(spec, head, { length, batch, markers, f16 });
          if (plan.buffers.some((b) => b.bytes > LIMITS.maxStorageBufferBindingSize)) { skipped += 1; continue; }
          built += 1;
          for (const seg of plan.segments) {
            for (const op of [...(seg.captureOps ?? []), ...seg.ops]) {
              for (const d of op.dispatch) {
                const n = d === 'rows' ? length * batch : d === 'rows8' ? Math.ceil(length * batch / 8) : d === 'rows16' ? Math.ceil(length * batch / 16) : d === 'rows32' ? Math.ceil(length * batch / 32) : d;
                if (n > LIMITS.maxComputeWorkgroupsPerDimension) {
                  over.push(`${id} L${length} B${batch} ${f16 ? 'f16' : 'f32'} ${seg.prefix}${op.name} (${op.kernel}): ${n}`);
                }
              }
            }
          }
        }
      }
    }
  }
  assert.ok(built > 1000 && skipped < built, `built ${built}, skipped ${skipped}`);
  assert.equal(over.length, 0, `${over.length} dispatches above 65535\n${over.slice(0, 8).join('\n')}`);
});
