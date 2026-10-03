// K28.R group B: errors reach the caller (R16), f16 range of the converter (R06,
// first half), f32-only capture (R19), finite outputs. The R16 and R06 tests are
// the review tests unchanged; R19 additionally expects the throw. The mock's
// error scopes are plain functions that tests replace to inject errors.
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { buildManifest, F16RangeError } from '../src/convert/manifest.ts';
import { Kleinhirn } from '../src/index.ts';
import { JuliaEngine } from '../src/julia.ts';
import { LruCache } from '../src/cache.ts';
import { getDevice, scopedCall } from '../src/device.ts';
import { assertFinite } from '../src/tasks.ts';
import { model, planOf, executeMock } from './helpers/review.mjs';
import { loadTiny } from './helpers/tiny-model.mjs';
import { createMockGpu } from './helpers/mock-gpu.ts';

const MINILM = 'sentence-transformers/all-MiniLM-L6-v2';

function withNavigator(t, gpu) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else delete globalThis.navigator;
  });
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: {
    requestAdapter: async () => ({ features: new Set(), info: {}, requestDevice: async () => gpu.device }),
  } } });
}

// Counts pushes and pops; `pop` decides what the pop of scope n resolves to.
function scopeSpy(gpu, pop = () => null) {
  const events = [];
  let n = 0;
  gpu.device.pushErrorScope = (filter) => { events.push(`push:${filter}`); };
  gpu.device.popErrorScope = () => { events.push('pop'); n += 1; return Promise.resolve(pop(n)); };
  const queue = gpu.device.queue;
  const write = queue.writeBuffer.bind(queue);
  const submit = queue.submit.bind(queue);
  queue.writeBuffer = (...a) => { events.push('write'); write(...a); };
  queue.submit = (...a) => { events.push('submit'); submit(...a); };
  for (const b of gpu.buffers) {
    const map = b.mapAsync.bind(b);
    b.mapAsync = async (...a) => { events.push('map'); return map(...a); };
  }
  return events;
}

test('R06 f16 converter must reject finite weights that become Inf', async () => {
  await assert.rejects(() => buildManifest([
    { name: 'head.classifier.bias', shape: [1], data: Float32Array.of(70000) },
  ], 'f16', {}), /overflow|finite|f16/i);
});

test('R06 converter: the f16 range, non-finite sources and the f32 manifest', async () => {
  const one = (data, dtype, name = 'head.classifier.bias') => buildManifest([{ name, shape: [data.length], data }], dtype, {});
  // 65504 is the largest f16; 65519 still rounds to it, 65520 rounds to Inf
  await one(Float32Array.of(65504, -65504, 65519, 0, 1e-8), 'f16');
  await assert.rejects(() => one(Float32Array.of(1, 65520), 'f16', 'layers.0.ffn_in.weight'),
    (e) => e instanceof F16RangeError && /layers\.0\.ffn_in\.weight: 65520 at index 1 overflows f16/.test(e.message));
  await assert.rejects(() => one(Float32Array.of(-1e9), 'f16'), F16RangeError);
  // the same data is a valid f32 manifest
  const built = await one(Float32Array.of(70000, -1e9), 'f32');
  assert.equal(built.manifest.tensors[0].dtype, 'f32');
  // NaN and Inf in the source: rejected in both dtypes, and not as an f16 range problem
  for (const dtype of ['f32', 'f16']) {
    for (const v of [NaN, Infinity, -Infinity]) {
      await assert.rejects(() => one(Float32Array.of(0, v), dtype, 'w'),
        (e) => !(e instanceof F16RangeError) && /w: source value .* at index 1 is not finite/.test(e.message));
    }
  }
  // the word table is checked per row chunk under its own name
  await assert.rejects(() => buildManifest([{ name: 'embeddings.word.weight', shape: [2, 1], data: Float32Array.of(1, 1e6) }], 'f16', {}),
    /embeddings\.word\.weight: 1000000 at index 1 overflows f16/);
});

test('R16 engine must install uncapturederror listener or validation scopes', async (t) => {
  const { engine, gpu } = model();
  let installed = false;
  gpu.device.addEventListener = (name) => { if (name === 'uncapturederror') installed = true; };
  gpu.device.pushErrorScope = () => { installed = true; };
  gpu.device.popErrorScope = async () => null;
  gpu.device.lost = new Promise(() => {});
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else delete globalThis.navigator;
  });
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: {
    requestAdapter: async () => ({ features: new Set(), info: {}, requestDevice: async () => gpu.device }),
  } } });
  // Real device helper, but ALL navigator/adapter/device calls here are mocks.
  engine.kh = await getDevice(false, true);
  // Actual WebGPU errors are asynchronous, not JS exceptions. A thrown mock would hide the bug.
  await engine.runIds({ inputIds: [0] });
  assert.ok(installed || typeof gpu.device.onuncapturederror === 'function',
    'neither scope nor uncapturederror listener installed');
});

test('R19 f16 capture must reject before issuing oversized buffer copies', () => {
  const p = planOf('sentence-transformers/all-MiniLM-L6-v2', 128, 1, true);
  const { gpu, executor } = executeMock(p);
  assert.throws(() => executor.submit(true), /capture needs an f32 plan/);
  const invalid = gpu.log.filter((e) => e.op === 'copy' && e.size + e.so > e.src.size);
  assert.equal(invalid.length, 0, `oversized capture copies=${invalid.length}`);
  assert.equal(gpu.log.filter((e) => e.op === 'copy').length, 0);
});

test('capture: f32 B1 only, in the executor and before the upload in all three engines', async () => {
  // executor: B>1 and an over-limit capture buffer
  const batched = executeMock(planOf(MINILM, 128, 4));
  assert.throws(() => batched.executor.submit(true), /capture needs a plan of batch 1, this one has 4/);
  const big = executeMock(planOf(MINILM, 128, 1));
  big.gpu.device.limits.maxBufferSize = 1000;
  assert.throws(() => big.executor.submit(true), /capture buffer needs \d+ B, maxBufferSize is 1000/);
  const fine = executeMock(planOf(MINILM, 128, 1));
  fine.executor.submit(true); // the frozen configuration: f32, B1
  // EncoderModel: f16 engine, nothing is written
  const { engine, gpu } = model(MINILM, 'f16', [128]);
  gpu.markBuilt();
  await assert.rejects(() => engine.runIds({ inputIds: [0] }, { capture: true }), /capture needs an f32 plan/);
  assert.equal(gpu.log.filter((e) => e.op === 'write' && e.phase === 'run').length, 0);
  // GLiNER and Julia: the plan of the bucket decides
  const f16 = executeMock(planOf(MINILM, 128, 1, true));
  f16.gpu.markBuilt();
  const kh = Reflect.construct(Kleinhirn, [{ device: f16.gpu.device }, {}, null, {}, 1, 'f16']);
  kh.plans.set(128, f16.executor);
  const input = { inputIds: Int32Array.of(0), seqLen: 1, markerIndices: Int32Array.of(), markerMask: Float32Array.of(), markerGroups: Int32Array.of() };
  await assert.rejects(() => kh.runPrepared(input, true, 128), /capture needs an f32 plan/);
  const julia = Reflect.construct(JuliaEngine, [{ device: f16.gpu.device }, {}, null, { options: 4 }, 'f16']);
  julia.plans.set(128, f16.executor);
  await assert.rejects(() => julia.runPrepared({ inputIds: Int32Array.of(0), markers: [], qtype: 0, seqLen: 1 }, true, 128),
    /capture needs an f32 plan/);
  assert.equal(f16.gpu.log.filter((e) => e.op === 'write' && e.phase === 'run').length, 0);
});

test('scope: push before upload, pop right after submit, readback after the pop, call settles after both', async () => {
  const { engine, gpu } = model(MINILM, 'f32', [128]);
  const events = scopeSpy(gpu);
  await engine.runIds({ inputIds: [0, 1] });
  assert.deepEqual(events.filter((e) => e !== 'write'), ['push:validation', 'submit', 'pop', 'map']);
  assert.ok(events.indexOf('push:validation') < events.indexOf('write'));
  assert.ok(events.lastIndexOf('write') < events.indexOf('submit'));
  // a batch call is one scoped call per piece
  events.length = 0;
  await engine.runIdsBatch([{ inputIds: [0] }, { inputIds: [1, 2] }]);
  assert.equal(events.filter((e) => e === 'push:validation').length, events.filter((e) => e === 'pop').length);
  assert.ok(events.indexOf('push:validation') < events.indexOf('submit') && events.indexOf('submit') < events.indexOf('pop'));
});

test('scope: the result waits for the scope, a validation error rejects the call', async () => {
  const { engine, gpu } = model(MINILM, 'f32', [128]);
  let release;
  scopeSpy(gpu, () => new Promise((resolve) => { release = resolve; }));
  let settled = false;
  const call = engine.runIds({ inputIds: [0] }).finally(() => { settled = true; });
  await new Promise((r) => setTimeout(r, 20)); // readback is long done
  assert.equal(settled, false);
  release({ message: 'Binding size (1) is larger than the maximum binding size (0)' });
  await assert.rejects(() => call, /WebGPU validation error: Binding size \(1\) is larger/);
  // same for a batch call and for the next call after an error (scopes were popped, the engine stays usable)
  const { engine: e2, gpu: g2 } = model(MINILM, 'f32', [128]);
  scopeSpy(g2, (n) => (n === 1 ? { message: 'bad pass' } : null));
  await assert.rejects(() => e2.runIdsBatch([{ inputIds: [0] }, { inputIds: [1] }]), /validation error: bad pass/);
  assert.equal((await e2.runIds({ inputIds: [0] })).rows, 1);
});

test('scope: an exception in the synchronous part still pops the scope, and the call throws it', async () => {
  const { engine, gpu } = model(MINILM, 'f32', [128]);
  const events = scopeSpy(gpu);
  await assert.rejects(() => engine.runIds({ inputIds: [999] }), /outside the vocabulary/);
  assert.deepEqual(events, ['push:validation', 'pop']);
  // a rejected pop promise of that scope must not become an unhandled rejection
  gpu.device.popErrorScope = () => Promise.reject(new Error('lost'));
  await assert.rejects(() => engine.runIds({ inputIds: [999] }), /outside the vocabulary/);
  await new Promise((r) => setTimeout(r, 10));
});

test('scopedCall: a read that rejects while the scope is clean rejects with the read error', async () => {
  const gpu = createMockGpu();
  const kh = { device: gpu.device };
  await assert.rejects(() => scopedCall(kh, () => 1, async () => { throw new Error('map failed'); }), /map failed/);
  assert.equal(await scopedCall(kh, () => 2, async (n) => n * 21), 42);
});

test('device: an uncaptured error and a lost device are kept and make every later call throw', async (t) => {
  const gpu = createMockGpu();
  let uncaptured;
  gpu.device.addEventListener = (name, fn) => { if (name === 'uncapturederror') uncaptured = fn; };
  let lose;
  gpu.device.lost = new Promise((resolve) => { lose = resolve; });
  withNavigator(t, gpu);
  const kh = await getDevice(false, true);
  assert.equal(kh.failure, null);
  const { engine } = model(MINILM, 'f32', [128]);
  engine.kh = kh;
  await engine.runIds({ inputIds: [0] }); // the model()'s own device is another mock; the scope calls go to gpu.device
  uncaptured({ error: { message: 'out of memory' } });
  await assert.rejects(() => engine.runIds({ inputIds: [0] }), /WebGPU error: out of memory/);
  await assert.rejects(() => engine.runIdsBatch([{ inputIds: [0] }]), /out of memory/); // first error stays
  // a lost device
  const gpu2 = createMockGpu();
  let lose2;
  gpu2.device.lost = new Promise((resolve) => { lose2 = resolve; });
  withNavigator(t, gpu2);
  const kh2 = await getDevice(false, true);
  lose2({ reason: 'destroyed', message: 'device was destroyed' });
  await new Promise((r) => setTimeout(r, 0));
  const { engine: e2 } = model(MINILM, 'f32', [128]);
  e2.kh = kh2;
  await assert.rejects(() => e2.runIds({ inputIds: [0] }), /WebGPU device lost \(destroyed\): device was destroyed/);
  void lose;
});

test('load: plan construction runs in a validation scope and a validation error fails the load', async () => {
  const gpu = createMockGpu();
  const events = scopeSpy(gpu);
  const ok = await loadTiny({ gpu });
  ok.restore();
  assert.deepEqual(events.filter((e) => e.startsWith('push') || e === 'pop'), ['push:validation', 'pop']);
  assert.ok(events.indexOf('push:validation') < events.indexOf('submit') || !events.includes('submit'));
  const bad = createMockGpu();
  scopeSpy(bad, () => ({ message: 'invalid buffer descriptor' }));
  await assert.rejects(() => loadTiny({ gpu: bad }), /WebGPU validation error: invalid buffer descriptor/);
});

test('finite outputs: NaN and Inf throw before softmax, argmax or the cache', async () => {
  assert.doesNotThrow(() => assertFinite(Float32Array.of(0, -1e4, 1e30), 'x'));
  assert.throws(() => assertFinite(Float32Array.of(0, NaN, Infinity, -Infinity), 'engine output'),
    /engine output: 3 of 4 values are not finite/);
  // EncoderModel: the readback of the staging buffer holds a NaN
  const { engine } = model(MINILM, 'f32', [128]);
  const staging = engine.plans.get(128).staging;
  const poisoned = (value) => () => new Float32Array(staging.size / 4).fill(value).buffer;
  staging.getMappedRange = poisoned(NaN);
  await assert.rejects(() => engine.runIds({ inputIds: [0] }), /engine output: \d+ of \d+ values are not finite/);
  staging.getMappedRange = poisoned(Infinity);
  await assert.rejects(() => engine.runIdsBatch([{ inputIds: [0] }]), /not finite/);
  staging.getMappedRange = poisoned(0.5);
  assert.equal((await engine.runIds({ inputIds: [0] })).data[0], 0.5);
  // GLiNER and Julia: nothing reaches the cache
  const plan = (v) => ({
    length: 128, markers: 4, batch: 1, upload() {}, submit() {}, assertCapturable() {},
    readLogits: async () => Float32Array.of(v, 0, 0, 0),
  });
  const device = createMockGpu().device;
  const kh = Reflect.construct(Kleinhirn, [{ device }, {}, null, {}, 1, 'f32']);
  kh.cache = new LruCache(4);
  kh.plans.set(128, plan(NaN));
  kh.embeddingRows = () => new Float32Array(1);
  const input = { inputIds: Int32Array.of(0), seqLen: 1, markerIndices: Int32Array.of(0), markerMask: Float32Array.of(1), markerGroups: Int32Array.of(0) };
  await assert.rejects(() => kh.runPrepared(input), /logits: 1 of 4 values are not finite/);
  assert.equal(kh.cache.size, 0);
  const julia = Reflect.construct(JuliaEngine, [{ device }, {}, null, { options: 4 }, 'f32']);
  julia.cache = new LruCache(4);
  julia.plans.set(128, plan(Infinity));
  julia.embeddingRows = () => new Float32Array(1);
  await assert.rejects(() => julia.runPrepared({ inputIds: Int32Array.of(0), markers: [0], qtype: 0, seqLen: 1 }),
    /logits: 1 of 4 values are not finite/);
  assert.equal(julia.cache.size, 0);
  // the masked logits of GLiNER and Julia are finite by construction
  assert.match(readFileSync('src/kernels/masklogits.wgsl', 'utf8'), /-1e4/);
});

test('mock: the no-op scope methods record nothing in the trace log', async () => {
  const gpu = createMockGpu();
  gpu.device.pushErrorScope('validation');
  assert.equal(await gpu.device.popErrorScope(), null);
  gpu.device.addEventListener('uncapturederror', () => {});
  assert.equal(gpu.log.length, 0);
  assert.ok(gpu.device.lost instanceof Promise);
});
