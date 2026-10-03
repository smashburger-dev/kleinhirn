// EncoderModel.load on a tiny in-memory model: fake fetch (manifest, one
// shard, tokenizer.json) and a navigator.gpu that hands out the recording
// mock. No network, no GPU. Weights are four bytes each, the plan only needs
// the names. The tokenizer is a stored fixture, so its byte count is known.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { EncoderModel } from '../../src/encoder.ts';
import { buildPlan } from '../../src/plan/build.ts';
import { specFromHfConfig } from '../../src/plan/hf.ts';
import { createMockGpu, installGpuGlobals } from './mock-gpu.ts';

installGpuGlobals();
const URL_BASE = 'http://tiny.test/models/tiny/f32/';
export const TOKENIZER_TEXT = readFileSync('tests/fixtures/hf-tokenizers/bpe-roberta.json', 'utf8');

export function tinyModel(overrides = {}) {
  const { spec, head, task } = specFromHfConfig({
    model_type: 'bert', hidden_size: 16, num_hidden_layers: 1, num_attention_heads: 2,
    intermediate_size: 32, vocab_size: 8, max_position_embeddings: 4096, ...overrides,
  }, { task: 'token-classification' });
  const plan = buildPlan(spec, head, { length: 8, batch: 1, markers: 0, f16: false });
  const names = new Set(plan.segments.flatMap((s) => [...(s.captureOps ?? []), ...s.ops])
    .flatMap((o) => o.bind).filter((id) => id.startsWith('w:')).map((id) => id.slice(2)));
  const parts = [];
  const tensors = [];
  let length = 0;
  const add = (entry, bytes) => {
    const pad = (256 - (length % 256)) % 256;
    parts.push(new Uint8Array(pad), bytes);
    length += pad;
    tensors.push({ ...entry, dtype: 'f32', shard: 0, offset: length, byteLength: bytes.length });
    length += bytes.length;
  };
  const words = new Float32Array(spec.vocab * spec.embeddingSize).fill(0.5);
  add({ name: 'embeddings.word.weight', shape: [spec.vocab, spec.embeddingSize],
    keepOnCpu: true, rowStart: 0, rowEnd: spec.vocab }, new Uint8Array(words.buffer));
  for (const name of names) add({ name, shape: [1] }, new Uint8Array(Float32Array.of(0.25).buffer));
  const shard = new Uint8Array(length);
  let off = 0;
  for (const p of parts) { shard.set(p, off); off += p.length; }
  const manifest = {
    format: 'kleinhirn-weights-2', version: 2, source: {}, encoder: {}, spec, head, task,
    labels: { 0: 'O', 1: 'X' }, tokenizer: 'tokenizer.json', maxLength: 128, tensors,
    shards: [{ file: 'weights-0.bin', bytes: shard.length,
      sha256: createHash('sha256').update(shard).digest('hex') }],
  };
  return { spec, head, manifest, shard };
}

// Installs the globals, loads, and returns { engine, gpu, restore, bytes }.
// bytes: the sizes the download counter must add up.
export async function loadTiny({ buckets = [128], overrides = {}, gpu = createMockGpu(), setup, tokenizer = true } = {}) {
  const { manifest, shard } = tinyModel(overrides);
  const manifestText = JSON.stringify(manifest);
  const files = new Map([
    [`${URL_BASE}manifest.json`, new TextEncoder().encode(manifestText)],
    [`${URL_BASE}weights-0.bin`, shard],
    ...(tokenizer ? [[`${URL_BASE}tokenizer.json`, new TextEncoder().encode(TOKENIZER_TEXT)]] : []),
  ]);
  const saved = { fetch: globalThis.fetch, navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator') };
  globalThis.fetch = async (url) => {
    const bytes = files.get(String(url));
    if (!bytes) return { ok: false, status: 404 };
    const text = new TextDecoder().decode(bytes);
    return {
      ok: true, status: 200, json: async () => JSON.parse(text), text: async () => text,
      arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    };
  };
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: {
    requestAdapter: async () => ({ features: new Set(), info: {}, requestDevice: async () => gpu.device }),
  } } });
  const restore = () => {
    globalThis.fetch = saved.fetch;
    if (saved.navigator) Object.defineProperty(globalThis, 'navigator', saved.navigator);
    else delete globalThis.navigator;
  };
  if (setup) setup(gpu);
  try {
    const engine = await EncoderModel.load({
      manifestUrl: `${URL_BASE}manifest.json`, precision: 'f32', buckets });
    return {
      engine, gpu, restore,
      bytes: {
        manifest: new TextEncoder().encode(manifestText).length, shard: shard.length,
        tokenizer: new TextEncoder().encode(TOKENIZER_TEXT).length,
        gpuWeights: manifest.tensors.filter((t) => !t.keepOnCpu).length * 4 },
    };
  } catch (e) {
    restore();
    throw e;
  }
}
