// K28.R group A: plan checker, batch halving, load-time rejections (R01, R02,
// R03, R08, R20). The R02 tests and R03 B16 are the review tests, unchanged;
// R01, R03 (L4096 single plan), R08 and R20 test the ordered contract (a
// rejection naming bucket, limit and need) instead of the kernel contracts, which
// stay red in tests/review/engine.test.mjs. The recording mock does not run WGSL.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdirSync } from 'node:fs';
import { buildPlan } from '../src/plan/build.ts';
import { WORKGROUP_BYTES, assertPlan, checkPlan, fitBatchPlan } from '../src/plan/check.ts';
import { specFromHfConfig } from '../src/plan/hf.ts';
import { loadWeights } from '../src/weights.ts';
import { model, planOf, source, MIN } from './helpers/review.mjs';
import { loadTiny } from './helpers/tiny-model.mjs';
import { createMockGpu } from './helpers/mock-gpu.ts';
import { pilotSpec, PILOT_IDS, PILOT_K285_IDS, PILOT_K286_IDS } from './helpers/pilot.ts';

const GRANITE = 'ibm-granite/granite-embedding-small-english-r2';
const MINILM = 'sentence-transformers/all-MiniLM-L6-v2';

// Only constructs plan metadata: the old fixture's CPU table is never read.
function bertPlanOwner(overrides = {}) {
  const { spec, head } = specFromHfConfig({
    model_type: 'bert', hidden_size: 64, num_hidden_layers: 1, num_attention_heads: 1,
    intermediate_size: 128, vocab_size: 8, max_position_embeddings: 8192,
    ...overrides,
  }, { task: 'token-classification' });
  const { engine } = model(MINILM, 'f32', []);
  engine.spec = spec;
  engine.head = head;
  return engine;
}

test('R02 GeGLU batch guard must split granite f32 B16 L1024', () => {
  const { engine } = model('ibm-granite/granite-embedding-small-english-r2', 'f32', [1024]);
  // Current guard accepts 96 MiB; actual mid has 192 MiB.
  const p = engine.batchPlan(1024, 16);
  assert.ok(p.plan.buffers.every((b) => b.bytes <= MIN.binding),
    JSON.stringify(p.plan.buffers.filter((b) => b.bytes > MIN.binding)));
});

test('R02 token head width must be included in the batch binding guard', () => {
  const engine = bertPlanOwner({ num_labels: 2048, max_position_embeddings: 2048 });
  const p = engine.batchPlan(2048, 16); // guard=24 MiB, out=256 MiB
  assert.ok(p.plan.buffers.every((b) => b.bytes <= MIN.binding),
    JSON.stringify(p.plan.buffers.filter((b) => b.bytes > MIN.binding)));
});

test('R02 convolution im2col width must be included in the batch binding guard', () => {
  const { engine } = model('antoinelouis/crossencoder-camemberta-L6-mmarcoFR', 'f32', []);
  // Same supported dimensions, valid odd kernel 7 instead of the frozen kernel 3.
  engine.spec = { ...engine.spec, conv: { kernel: 7, act: 'gelu' } };
  const p = engine.batchPlan(512, 16); // guard=96 MiB, cols=168 MiB
  assert.ok(p.plan.buffers.every((b) => b.bytes <= MIN.binding),
    JSON.stringify(p.plan.buffers.filter((b) => b.bytes > MIN.binding)));
});

test('R03 B16 L4096 must not dispatch 65536 rows', () => {
  const p = bertPlanOwner().batchPlan(4096, 16); // guard=48 MiB: public batch guard accepts it
  const op = p.plan.segments[0].ops[0];
  assert.equal(op.dispatch[0], 'rows');
  assert.ok(p.length * p.batch <= MIN.groups);
});

test('batch halving: sizes tried are the BATCH_SIZES below the request, largest first', () => {
  const built = [];
  const fits = (batch) => { built.push(batch); return planOf(GRANITE, 1024, batch); };
  const plan = fitBatchPlan(fits, {}, 16);
  assert.equal(plan.batch, 8);
  assert.deepEqual(built, [16, 8]); // B16 rejected (mid 192 MiB), B8 fits (96 MiB)
  assert.equal(fitBatchPlan((b) => planOf(GRANITE, 1024, b), {}, 4).batch, 4);
  // nothing fits: a head whose output is wider than the binding allows at B4
  const owner = bertPlanOwner({ num_labels: 70000, max_position_embeddings: 128 });
  assert.equal(owner.batchPlan(128, 16), undefined);
  assert.equal(owner.batchPlan(128, 4), undefined);
  assert.equal(owner.batchPlans.size, 0); // no plan was built
});

test('batch halving: a chunk that fits no batch plan runs on the bucket plan, one row each', async () => {
  // stride 64 for these rows: B4 out = 4 * 64 * 140000 * 4 B = 143 MB > 128 MiB, B1 out = 72 MB
  const owner = bertPlanOwner({ num_labels: 140000, max_position_embeddings: 128 });
  owner.plans.set(128, owner.makePlan(128, 1));
  const before = owner.plans.get(128).gpuBytes;
  const rows = [[1, 2, 3], [1, 2], [1, 2, 3, 4], [1]].map((ids) => ({ inputIds: ids }));
  const out = await owner.runIdsBatch(rows);
  assert.deepEqual(out.map((o) => o.seqLen), [3, 2, 4, 1]);
  assert.deepEqual(out.map((o) => o.rows), [3, 2, 4, 1]);
  assert.equal(owner.batchPlans.size, 0);
  assert.equal(owner.plans.get(128).gpuBytes, before);
});

test('batch halving: a halved chunk runs in pieces of the fitted size and keeps the order', async () => {
  const { engine } = model(GRANITE, 'f32', [1024]);
  const rows = Array.from({ length: 16 }, (_, i) => ({ inputIds: new Int32Array(900 + i) }));
  const out = await engine.runIdsBatch(rows);
  assert.deepEqual(out.map((o) => o.seqLen), rows.map((r) => r.inputIds.length));
  assert.deepEqual([...engine.batchPlans.keys()], ['960:8']); // stride 960, B16 never built, two pieces of B8
});

test('checkPlan: every shipped pilot plan fits the minimum limits at the default buckets', () => {
  for (const id of [...PILOT_IDS, ...PILOT_K285_IDS, ...PILOT_K286_IDS]) {
    const { spec, head } = pilotSpec(id);
    for (const length of [128, 512]) {
      const plan = buildPlan(spec, head, { length, batch: 1, markers: 0, f16: false });
      assert.deepEqual(checkPlan(plan, {}), [], `${id} L${length}`);
    }
  }
});

test('R01: a bucket length outside multiples of 4 is planned (remainder loops), a length that is no positive integer is rejected', async () => {
  for (const length of [0, -4, 6.5]) {
    const plan = planOf(MINILM, 128, 1);
    plan.length = length;
    const bad = checkPlan(plan, {});
    assert.ok(bad.some((m) => m.includes(`bucket L${length} B1`) && /positive integer/.test(m)), `${length}: ${bad}`);
  }
  for (const length of [5, 130]) assert.deepEqual(checkPlan(planOf(MINILM, length, 1), {}), [], `L${length}`);
  await assert.doesNotReject(() => loadTiny({ buckets: [130] }));
});

test('R03 L4100: a single plan over the workgroup memory limit is rejected at load, with need and limit', async () => {
  // L4100 is no multiple of 32, so attention takes the per-row kernel: 4 * 4100 + 256 = 16656 bytes.
  const bad = checkPlan(planOf(GRANITE, 4100, 1), {});
  assert.ok(bad.some((m) => m.includes('bucket L4100 B1')
    && m.includes('maxComputeWorkgroupStorageSize is 16384') && m.includes('needs 16656')), `${bad}`);
  // the flash kernel (K27) needs 16 KiB at any multiple of 32: L4096 fits the minimum now
  assert.deepEqual(checkPlan(planOf(GRANITE, 4096, 1), {}), []);
  assert.deepEqual(checkPlan(planOf(GRANITE, 4100, 1), { maxComputeWorkgroupStorageSize: 32768 }), []);
  await assert.rejects(() => loadTiny({ buckets: [4100] }),
    /bucket L4100 B1.*workgroup memory.*needs 16656.*maxComputeWorkgroupStorageSize is 16384/);
});

test('R03 grid: both dispatch axes are checked against maxComputeWorkgroupsPerDimension', () => {
  const plan = bertPlanOwner().planFor(4096, 16);
  const bad = checkPlan(plan, {}, undefined, { batchOnly: true });
  assert.ok(bad.some((m) => m.includes('bucket L4096 B16') && m.includes('needs 65536')
    && m.includes('maxComputeWorkgroupsPerDimension is 65535')), `${bad}`);
  // the second axis: 'rows' in dispatch[1], and the rows16 form
  const synthetic = planOf(MINILM, 128, 1);
  synthetic.segments[0].ops[0].dispatch = [1, 'rows16'];
  synthetic.length = 1048576 + 4;
  assert.ok(checkPlan(synthetic, {}).some((m) => /needs 65537.*maxComputeWorkgroupsPerDimension/.test(m)));
});

test('R03 weights: a bound weight over the binding limit and a tensor over maxBufferSize are rejected', async () => {
  const plan = planOf(MINILM, 128, 1);
  const bad = checkPlan(plan, {}, (name) => (name === 'layers.0.qkv.weight' ? 201_326_592 : 4));
  assert.ok(bad.some((m) => m.includes('weight layers.0.qkv.weight needs 201326592')
    && m.includes('maxStorageBufferBindingSize is 134217728')), `${bad}`);
  assert.throws(() => assertPlan(plan, {}, () => 201_326_592), /bucket L128 B1: weight/);
  // loadWeights: the oversized tensor throws before any buffer is created
  const gpu = createMockGpu();
  gpu.device.limits.maxBufferSize = 1000;
  const manifest = {
    format: 'kleinhirn-weights-2', source: {}, encoder: {}, head: { type: 'x', temperature: 1 },
    tensors: [
      { name: 'small', dtype: 'f32', shape: [1], shard: 0, offset: 0, byteLength: 4 },
      { name: 'huge.weight', dtype: 'f32', shape: [300], shard: 0, offset: 0, byteLength: 1200 },
    ],
    shards: [{ file: 'w.bin', bytes: 4, sha256: createHash('sha256').update(Uint8Array.of(1, 2, 3, 4)).digest('hex') }],
  };
  const savedFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, status: 200, arrayBuffer: async () => Uint8Array.of(1, 2, 3, 4).buffer });
  try {
    await assert.rejects(() => loadWeights(gpu.device, 'http://x/m/manifest.json', manifest),
      /tensor huge\.weight needs 1200 B, maxBufferSize is 1000/);
  } finally { globalThis.fetch = savedFetch; }
  assert.equal(gpu.buffers.length, 0);
});

test('checkPlan: bindings per stage, staging and a missing limit counting as the spec minimum', () => {
  const plan = planOf(MINILM, 128, 1);
  plan.segments[0].ops[0].bind.push(...Array(9).fill('mask'));
  assert.ok(checkPlan(plan, {}).some((m) => /maxStorageBuffersPerShaderStage is 8/.test(m)));
  assert.deepEqual(checkPlan(plan, { maxStorageBuffersPerShaderStage: 32 }), []);
  const wide = planOf(MINILM, 128, 1);
  wide.output.bytes = 300_000_000;
  assert.ok(checkPlan(wide, {}).some((m) => /output staging buffer needs 300000000.*maxBufferSize is 268435456/.test(m)));
  assert.deepEqual(checkPlan(wide, { maxBufferSize: 400_000_000 }), []);
});

test('workgroup bytes of the checker equal the var<workgroup> declarations of every kernel', () => {
  const files = readdirSync('src/kernels').filter((f) => f.endsWith('.wgsl'));
  assert.deepEqual(files.map((f) => f.replace('.wgsl', '')).sort(), Object.keys(WORKGROUP_BYTES).sort());
  for (const f of files) {
    const code = source(`src/kernels/${f}`);
    // scalar elements count 4 bytes, vec4 elements 16 (the f32 plan, the larger one)
    const decls = [...code.matchAll(/var<workgroup>\s+\w+:\s*array<(f32|f16|u32|i32|vec4<(?:f32|f16|\{\{F\}\})>),\s*(\w+)>/g)];
    // scalar workgroup variables (K27 tile flags) count 4 bytes each
    const scalars = [...code.matchAll(/var<workgroup>\s+\w+:\s*(?:u32|i32|f32);/g)].length;
    const c = { N: 768 };
    for (const L of [4, 128, 1024, 4032]) {
      const bytes = 4 * scalars + decls.reduce((n, [, type, size]) => n + (type.startsWith('vec4') ? 16 : 4) * (size === 'L' ? L : size === 'N' ? c.N : Number(size)), 0);
      assert.equal(WORKGROUP_BYTES[f.replace('.wgsl', '')](L, c), bytes, `${f} at L${L}`);
    }
  }
});

test('R08: relative attention with a head width outside groups of 4 is planned (scalar remainder in attention.wgsl)', () => {
  const relative = (hidden) => specFromHfConfig({
    model_type: 'deberta-v2', hidden_size: hidden, num_hidden_layers: 1, num_attention_heads: 2,
    intermediate_size: 128, vocab_size: 8, max_position_embeddings: 512,
    relative_attention: true, share_att_key: true, norm_rel_ebd: 'layer_norm',
    position_buckets: 256, pos_att_type: ['p2c', 'c2p'], position_biased_input: false,
  }, { task: 'token-classification' });
  const at = (hidden) => {
    const { spec, head } = relative(hidden);
    return buildPlan(spec, head, { length: 128, batch: 1, markers: 0, f16: false });
  };
  assert.deepEqual(checkPlan(at(100), {}), []); // D = 50
  assert.deepEqual(checkPlan(at(128), {}), []); // D = 64
  // standard attention (mbattention) with D = 50 (polyBERT)
  const { spec, head } = pilotSpec('xushijie/polyBERT');
  assert.equal(spec.headDim % 4, 2);
  assert.deepEqual(checkPlan(buildPlan(spec, head, { length: 128, batch: 1, markers: 0, f16: false }), {}), []);
});

test('R20: an upload that is not a multiple of 4 bytes is rejected before any write', () => {
  const { spec, head } = specFromHfConfig({
    model_type: 'bert', hidden_size: 3, num_hidden_layers: 1, num_attention_heads: 1,
    intermediate_size: 6, vocab_size: 8,
  }, { task: 'token-classification' });
  const plan = buildPlan(spec, head, { length: 5, batch: 1, markers: 0, f16: true });
  assert.equal(plan.embeddingSize, 3);
  const bad = checkPlan(plan, {});
  assert.ok(bad.some((m) => /word row upload of 30 B is not a multiple of 4/.test(m)), `${bad}`);
  assert.ok(checkPlan(plan, {}, undefined, { batchOnly: true }).some((m) => /upload of 30 B/.test(m)));
  // an aligned length of the same shape is fine
  assert.deepEqual(checkPlan(buildPlan(spec, head, { length: 8, batch: 1, markers: 0, f16: true }), {}), []);
});
