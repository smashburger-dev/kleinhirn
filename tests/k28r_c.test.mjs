// K28.R group C: plan lifetime and ownership (R04, R26, R28 and the tokenizer
// bytes of EncoderModel). R04, R26 and R28 are the review tests, unchanged; the
// others cover the queue (no nested enqueue), gpuBytes from live executors and
// the cache copies of the Julia engine. The recording mock does not run WGSL.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Kleinhirn } from '../src/index.ts';
import { JuliaEngine } from '../src/julia.ts';
import { LruCache, juliaMergeKey, schemaMergeKey } from '../src/cache.ts';
import { model } from './helpers/review.mjs';
import { loadTiny } from './helpers/tiny-model.mjs';

const GRANITE = 'ibm-granite/granite-embedding-small-english-r2';
function within(promise, ms, what) {
  let timer;
  const limit = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what}: still pending after ${ms} ms (queue deadlock)`)), ms);
  });
  return Promise.race([promise, limit]).finally(() => clearTimeout(timer));
}

test('R04 nine concurrent batch calls must not evict a still queued plan', async () => {
  const { engine, destroyedWrites } = model('ibm-granite/granite-embedding-small-english-r2', 'f32', [1024]);
  // All nine batchPlans are created synchronously, before the first queue.then starts.
  const calls = Array.from({ length: 9 }, (_, i) => engine.runIdsBatch([
    { inputIds: new Int32Array(64 * (i + 1)) }, { inputIds: new Int32Array(64 * (i + 1)) },
  ]));
  await Promise.all(calls);
  assert.equal(destroyedWrites.length, 0, `uploads after destroy=${destroyedWrites.length}`);
});

test('R26 gpuBytes must subtract evicted batch plan buffers', () => {
  const { engine } = model('ibm-granite/granite-embedding-small-english-r2', 'f32', []);
  for (let i = 1; i <= 9; i += 1) engine.batchPlan(64 * i, 4);
  const activeBytes = [...engine.batchPlans.values()].reduce((n, p) => n + p.gpuBytes, 0);
  assert.equal(engine.info().gpuBytes, activeBytes);
});

test('R28 cached prepared results must not be corrupted by caller mutation', async () => {
  const engine = Reflect.construct(Kleinhirn, [{}, {}, null, {}, 1, 'f32']);
  engine.plans.set(128, { length: 128, markers: 16 });
  engine.cache = new LruCache(2);
  const input = { inputIds: Int32Array.of(0), seqLen: 1,
    markerIndices: Int32Array.of(0), markerMask: Float32Array.of(1), markerGroups: Int32Array.of(0) };
  engine.cache.set(schemaMergeKey(input), { logits: Float32Array.of(1), probabilities: Float32Array.of(1) });
  const first = await engine.runPrepared(input);
  first.logits[0] = 99;
  assert.equal((await engine.runPrepared(input)).logits[0], 1);
});

test('queue: eviction and plan building run in the queue, the plan map stays capped, no call hangs', async () => {
  const { engine, destroyedWrites } = model(GRANITE, 'f32', [1024]);
  const calls = Array.from({ length: 12 }, (_, i) => engine.runIdsBatch(
    Array.from({ length: 16 }, () => ({ inputIds: new Int32Array(40 * (i + 1)) }))));
  const outs = await within(Promise.all(calls), 5000, 'twelve concurrent B16 calls');
  assert.equal(outs.length, 12);
  assert.equal(destroyedWrites.length, 0);
  assert.ok(engine.batchPlans.size <= 8);
  // a call that needs two pieces (B16 -> 2 x B8 at stride 960) returns in input order
  const lens = Array.from({ length: 16 }, (_, i) => 900 + i);
  const out = await within(engine.runIdsBatch(lens.map((n) => ({ inputIds: new Int32Array(n) }))), 5000, 'two pieces');
  assert.deepEqual(out.map((o) => o.seqLen), lens);
});

test('gpuBytes: weights plus live bucket and batch plans, a capture buffer counts once it exists', async () => {
  const { engine } = model(GRANITE, 'f32', [128]);
  const live = () => engine.plans.get(128).gpuBytes
    + [...engine.batchPlans.values()].reduce((n, p) => n + p.gpuBytes, 0);
  assert.equal(engine.info().gpuBytes, live());
  const before = engine.info().gpuBytes;
  const plan = engine.plans.get(128);
  await engine.runIds({ inputIds: [0, 0, 0] }, { capture: true });
  assert.equal(engine.info().gpuBytes, before + plan.plan.captureSlots * plan.plan.captureSlotBytes);
  assert.equal(engine.info().gpuBytes, live());
  engine.batchPlan(64, 4);
  engine.batchPlan(128, 4);
  assert.equal(engine.info().gpuBytes, live());
});

test('EncoderModel.load counts the tokenizer file in downloadBytes, gpuBytes starts from the weights', async (t) => {
  const { engine, restore, bytes } = await loadTiny();
  t.after(restore);
  const info = engine.info();
  assert.equal(info.downloadBytes, bytes.manifest + bytes.shard + bytes.tokenizer);
  assert.equal(info.gpuBytes, bytes.gpuWeights + engine.plans.get(128).gpuBytes);
  assert.equal(info.hasTokenizer, true);
});

test('EncoderModel.load: a missing tokenizer file counts no bytes and keeps the id path usable', async (t) => {
  const { restore, engine, bytes } = await loadTiny({ tokenizer: false });
  t.after(restore);
  assert.equal(engine.info().hasTokenizer, false);
  assert.equal(engine.info().downloadBytes, bytes.manifest + bytes.shard);
  await assert.rejects(() => engine.tokenClassify('x'), /text input unavailable|no tokenizer/);
});

for (const [name, make] of [
  ['Julia', () => Reflect.construct(JuliaEngine, [{}, {}, null, { options: 4 }, 'f32']), ],
]) {
  test(`${name}: cache hits and batch duplicates get their own arrays`, async () => {
    const engine = make();
    engine.plans.set(128, { length: 128, markers: 4 });
    engine.cache = new LruCache(4);
    const input = { inputIds: Int32Array.of(5), markers: [0], qtype: 0, seqLen: 1 };
    engine.cache.set(juliaMergeKey(input), { logits: Float32Array.of(1), probabilities: Float32Array.of(1) });
    const first = await engine.runPrepared(input);
    first.logits[0] = 99;
    first.probabilities[0] = 99;
    assert.equal((await engine.runPrepared(input)).logits[0], 1);
    assert.equal((await engine.runPrepared(input)).probabilities[0], 1);
    // a batch of a cached row and of two identical uncached rows
    const other = { inputIds: Int32Array.of(6), markers: [0], qtype: 0, seqLen: 1 };
    engine.runBatchChunk = async (rows) => rows.map(() => ({ logits: Float32Array.of(7), probabilities: Float32Array.of(1) }));
    const [hit, a, b] = await engine.runPreparedBatch([input, other, other]);
    assert.equal(hit.logits[0], 1);
    a.logits[0] = 99;
    assert.equal(b.logits[0], 7);
    hit.logits[0] = 99;
    assert.equal((await engine.runPrepared(input)).logits[0], 1);
  });
}

test('Kleinhirn: batch duplicates and cache hits get their own arrays', async () => {
  const engine = Reflect.construct(Kleinhirn, [{}, {}, null, {}, 1, 'f32']);
  engine.plans.set(128, { length: 128, markers: 16 });
  engine.cache = new LruCache(4);
  const row = (id) => ({ inputIds: Int32Array.of(id), seqLen: 1,
    markerIndices: Int32Array.of(0), markerMask: Float32Array.of(1), markerGroups: Int32Array.of(0) });
  engine.cache.set(schemaMergeKey(row(1)), { logits: Float32Array.of(1), probabilities: Float32Array.of(1) });
  engine.runBatchChunk = async (rows) => rows.map(() => ({ logits: Float32Array.of(7), probabilities: Float32Array.of(1) }));
  const [hit, a, b] = await engine.runPreparedBatch([row(1), row(2), row(2)]);
  a.logits[0] = 99;
  hit.logits[0] = 99;
  assert.equal(b.logits[0], 7);
  assert.equal((await engine.runPrepared(row(1))).logits[0], 1);
});
