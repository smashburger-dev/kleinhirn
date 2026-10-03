// K28.2 step 4: converter core unit tests (no model files needed): safetensors
// dtypes, the name table's error paths, and the shard writer.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { specFromHfConfig } from '../src/plan/hf.ts';
import { buildManifest, gatherTensors, sinusoidalTable, SHARD_LIMIT } from '../src/convert/manifest.ts';
import { detectPrefix, planNames } from '../src/convert/names.ts';
import { parseSafetensors, readF32 } from '../src/convert/safetensors.ts';

function safetensors(tensors: Record<string, { dtype: string; shape: number[]; data: Uint8Array }>): Uint8Array {
  const header: Record<string, unknown> = {};
  let off = 0;
  for (const [name, t] of Object.entries(tensors)) {
    header[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + t.data.length] };
    off += t.data.length;
  }
  const h = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + h.length + off);
  new DataView(out.buffer).setBigUint64(0, BigInt(h.length), true);
  out.set(h, 8);
  let p = 8 + h.length;
  for (const t of Object.values(tensors)) {
    out.set(t.data, p);
    p += t.data.length;
  }
  return out;
}
const bytes = (a: ArrayBufferView): Uint8Array => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

test('safetensors: F32, F16, BF16 and F64 read as f32, integers refuse', () => {
  const st = parseSafetensors(safetensors({
    a: { dtype: 'F32', shape: [2], data: bytes(new Float32Array([1.5, -2])) },
    b: { dtype: 'F16', shape: [2], data: bytes(new Float16Array([0.5, 65504])) },
    c: { dtype: 'BF16', shape: [2], data: bytes(new Uint16Array([0x3fc0, 0xc000])) }, // 1.5, -2
    d: { dtype: 'F64', shape: [1], data: bytes(new Float64Array([0.1])) },
    e: { dtype: 'I64', shape: [1], data: bytes(new BigInt64Array([7n])) },
  }));
  assert.deepEqual([...readF32(st, 'a')], [1.5, -2]);
  assert.deepEqual([...readF32(st, 'b')], [0.5, 65504]);
  assert.deepEqual([...readF32(st, 'c')], [1.5, -2]);
  assert.equal(readF32(st, 'd')[0], Math.fround(0.1));
  assert.throws(() => readF32(st, 'e'), /dtype I64/);
});

const tiny = { model_type: 'bert', hidden_size: 4, num_attention_heads: 2, num_hidden_layers: 1,
  intermediate_size: 8, vocab_size: 5, max_position_embeddings: 6, type_vocab_size: 2,
  id2label: { 0: 'a', 1: 'b' } };
const { spec, head } = specFromHfConfig(tiny, { task: 'sequence-classification' });
const shapes: Record<string, number[]> = {
  'embeddings.word_embeddings.weight': [5, 4], 'embeddings.position_embeddings.weight': [6, 4],
  'embeddings.token_type_embeddings.weight': [2, 4], 'embeddings.LayerNorm.weight': [4],
  'embeddings.LayerNorm.bias': [4],
  'encoder.layer.0.attention.self.query.weight': [4, 4], 'encoder.layer.0.attention.self.query.bias': [4],
  'encoder.layer.0.attention.self.key.weight': [4, 4], 'encoder.layer.0.attention.self.key.bias': [4],
  'encoder.layer.0.attention.self.value.weight': [4, 4], 'encoder.layer.0.attention.self.value.bias': [4],
  'encoder.layer.0.attention.output.dense.weight': [4, 4], 'encoder.layer.0.attention.output.dense.bias': [4],
  'encoder.layer.0.attention.output.LayerNorm.weight': [4], 'encoder.layer.0.attention.output.LayerNorm.bias': [4],
  'encoder.layer.0.intermediate.dense.weight': [8, 4], 'encoder.layer.0.intermediate.dense.bias': [8],
  'encoder.layer.0.output.dense.weight': [4, 8], 'encoder.layer.0.output.dense.bias': [4],
  'encoder.layer.0.output.LayerNorm.weight': [4], 'encoder.layer.0.output.LayerNorm.bias': [4],
  'pooler.dense.weight': [4, 4], 'pooler.dense.bias': [4],
};
const withPrefix = (prefix: string, head: Record<string, number[]>, drop: string[] = []): string[] =>
  [...Object.keys(shapes).map((n) => n.startsWith('pooler') ? `${prefix}${n}` : `${prefix}${n}`), ...Object.keys(head)]
    .filter((n) => !drop.includes(n));
const headNames = { 'classifier.weight': [2, 4], 'classifier.bias': [2] };

test('names: prefix detected once, known-unused tensors are skipped', () => {
  for (const prefix of ['bert.', '']) {
    const names = [...withPrefix(prefix, headNames), `${prefix}embeddings.position_ids`, 'cls.predictions.bias'];
    assert.equal(detectPrefix(names), prefix);
    const plan = planNames(spec, head, names);
    assert.equal(plan.prefix, prefix);
    assert.deepEqual(plan.unused.sort(), ['cls.predictions.bias', `${prefix}embeddings.position_ids`].sort());
    assert.equal(plan.tensors.find((t) => t.name === 'layers.0.qkv.weight')!.sources.length, 3);
  }
});

test('names: gamma and beta resolve to LayerNorm weight and bias', () => {
  const names = withPrefix('bert.', headNames).map((n) =>
    n.includes('LayerNorm') ? n.replace(/\.weight$/, '.gamma').replace(/\.bias$/, '.beta') : n);
  const plan = planNames(spec, head, names);
  assert.ok(plan.tensors.find((t) => t.name === 'embeddings.LayerNorm.weight')!.sources[0].endsWith('.gamma'));
});

test('names: unknown tensor, missing tensor and two prefixes are errors', () => {
  const base = withPrefix('bert.', headNames);
  assert.throws(() => planNames(spec, head, [...base, 'bert.encoder.layer.0.mystery.weight']),
    /bert\.encoder\.layer\.0\.mystery\.weight is neither used nor known/);
  assert.throws(() => planNames(spec, head, [...base, 'model.extra.weight']), /neither used nor known/);
  assert.throws(() => planNames(spec, head, base.filter((n) => n !== 'bert.pooler.dense.weight')),
    /lacks required tensor bert\.pooler\.dense\.weight/);
  assert.throws(() => planNames(spec, head, [...base, 'embeddings.word_embeddings.weight']),
    /cannot detect the encoder prefix/);
});

test('gather fuses Q, K, V by rows and checks shapes', () => {
  const names = withPrefix('bert.', headNames);
  const entries: Record<string, { dtype: string; shape: number[]; data: Uint8Array }> = {};
  let k = 0;
  const all: Record<string, number[]> = {};
  for (const n of names) all[n] = shapes[n.replace('bert.', '')] ?? headNames[n as keyof typeof headNames];
  for (const [n, shape] of Object.entries(all)) {
    const count = shape.reduce((a, b) => a * b, 1);
    entries[n] = { dtype: 'F32', shape, data: bytes(new Float32Array(count).fill(k += 1)) };
  }
  const st = parseSafetensors(safetensors(entries));
  const plan = planNames(spec, head, names);
  const tensors = gatherTensors(st, plan);
  const qkv = tensors.find((t) => t.name === 'layers.0.qkv.weight')!;
  assert.deepEqual(qkv.shape, [12, 4]);
  const q = entries['bert.encoder.layer.0.attention.self.query.weight'];
  const v = entries['bert.encoder.layer.0.attention.self.value.weight'];
  assert.equal(qkv.data[0], new Float32Array(q.data.buffer, q.data.byteOffset, 1)[0]);
  assert.equal(qkv.data[qkv.data.length - 1], new Float32Array(v.data.buffer, v.data.byteOffset, 1)[0]);
  // a wrong vocabulary size in the description is caught
  const bad = specFromHfConfig({ ...tiny, vocab_size: 6 }, { task: 'sequence-classification' });
  assert.throws(() => gatherTensors(st, planNames(bad.spec, bad.head, names)), /do not fit/);
});

test('shards: alignment 256, split at the limit, embedding rows chunked', async () => {
  const meta = { source: { repo: 'x', revision: 'y', checkpointSha256: 'z' }, spec, head,
    task: 't', labels: {}, tokenizer: 'tokenizer.json', maxLength: 6 };
  const big = new Float32Array(SHARD_LIMIT / 2 - 500); // nearly a full shard in f16
  const tensors = [
    { name: 'embeddings.word.weight', shape: [130_000, 1], data: new Float32Array(130_000) },
    { name: 'a', shape: [3], data: new Float32Array([1, 2, 3]) },
    { name: 'big', shape: [big.length], data: big },
    { name: 'b', shape: [2], data: new Float32Array([4, 5]) },
  ];
  const { manifest, shards } = await buildManifest(tensors, 'f16', meta);
  const entries = manifest.tensors as { name: string; shard: number; offset: number; rowStart?: number; rowEnd?: number; byteLength: number }[];
  assert.deepEqual(entries.filter((e) => e.name === 'embeddings.word.weight').map((e) => [e.rowStart, e.rowEnd]),
    [[0, 60_000], [60_000, 120_000], [120_000, 130_000]]);
  assert.ok(entries.every((e) => e.offset % 256 === 0));
  assert.ok(shards.every((s) => s.bytes.length <= SHARD_LIMIT));
  assert.ok(entries.find((e) => e.name === "big")!.shard > entries.find((e) => e.name === "a")!.shard);
  assert.ok(shards.length >= 2);
});

// K28.5: RoBERTa, XLM-R and DistilBERT name tables, generated position table,
// Dense modules.
const tinyRoberta = { ...tiny, model_type: 'roberta', max_position_embeddings: 8, type_vocab_size: 1,
  pad_token_id: 1, bos_token_id: 0 };
const robertaNames = (prefix: string): string[] => [
  ...Object.keys(shapes).filter((n) => !n.startsWith('pooler')).map((n) => `${prefix}${n}`),
  `${prefix}pooler.dense.weight`, `${prefix}pooler.dense.bias`,
  'classifier.dense.weight', 'classifier.dense.bias',
  'classifier.out_proj.weight', 'classifier.out_proj.bias',
];

test('names: RoBERTa head, prefix and unused LM head', () => {
  const r = specFromHfConfig(tinyRoberta, { task: 'sequence-classification', template: [0, null, 2] });
  assert.equal(r.spec.embed.positionOffset, 2);
  for (const prefix of ['roberta.', '']) {
    const names = [...robertaNames(prefix), 'lm_head.bias', 'lm_head.dense.weight'];
    const plan = planNames(r.spec, r.head, names);
    assert.equal(plan.prefix, prefix);
    assert.ok(plan.unused.includes('lm_head.bias'));
    assert.ok(plan.unused.includes(`${prefix}pooler.dense.weight`));
    assert.deepEqual(plan.tensors.filter((t) => t.name.startsWith('head.')).map((t) => t.sources[0]),
      ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']);
    // a bert. prefix is not a RoBERTa prefix
    if (prefix) {
      assert.throws(() => planNames(r.spec, r.head, names.map((n) => n.replace(/^roberta\./, 'bert.'))),
        /cannot detect the encoder prefix/);
    }
  }
});

test('RoBERTa offset: <s> equal to pad counts as padding', () => {
  const off = (c: Record<string, unknown>) =>
    specFromHfConfig({ ...tinyRoberta, ...c }, { task: 'sequence-classification',
      template: [c.first as number ?? 0, null, 2] }).spec.embed.positionOffset;
  assert.equal(off({}), 2);
  assert.equal(off({ pad_token_id: 0, first: 0 }), 0);
  assert.equal(off({ pad_token_id: 0, first: 1 }), 1);
});

const tinyDistil = { model_type: 'distilbert', dim: 4, n_heads: 2, n_layers: 1, hidden_dim: 8,
  vocab_size: 5, max_position_embeddings: 6, activation: 'gelu', sinusoidal_pos_embds: true,
  id2label: { 0: 'a', 1: 'b' } };
const distilNames = (prefix: string, withPosition: boolean): string[] => {
  const b = `${prefix}transformer.layer.0`;
  const w = (n: string) => [`${n}.weight`, `${n}.bias`];
  return [
    `${prefix}embeddings.word_embeddings.weight`,
    ...(withPosition ? [`${prefix}embeddings.position_embeddings.weight`] : []),
    ...w(`${prefix}embeddings.LayerNorm`),
    ...['q_lin', 'k_lin', 'v_lin', 'out_lin'].flatMap((k) => w(`${b}.attention.${k}`)),
    ...w(`${b}.sa_layer_norm`), ...w(`${b}.ffn.lin1`), ...w(`${b}.ffn.lin2`), ...w(`${b}.output_layer_norm`),
    ...w('pre_classifier'), ...w('classifier'),
    ...w('vocab_transform'), ...w('vocab_layer_norm'), ...w('vocab_projector'),
  ];
};

test('names: DistilBERT tensors, prefix, MLM head unused, sinusoidal table', () => {
  const d = specFromHfConfig(tinyDistil, { task: 'sequence-classification' });
  assert.equal(d.spec.embed.typeVocab, 0);
  assert.equal(d.spec.embed.sinusoidal, true);
  for (const prefix of ['distilbert.', '']) {
    const plan = planNames(d.spec, d.head, distilNames(prefix, true));
    assert.equal(plan.prefix, prefix);
    assert.ok(!plan.tensors.some((t) => t.name === 'embeddings.type.weight'));
    assert.ok(plan.tensors.find((t) => t.name === 'embeddings.position.weight')!.sources.length === 1);
    assert.ok(plan.unused.includes('vocab_transform.weight'));
    assert.equal(plan.tensors.find((t) => t.name === 'layers.0.qkv.weight')!.sources
      .map((s) => s.split('.').at(-2)).join(), 'q_lin,k_lin,v_lin');
    assert.ok(plan.tensors.some((t) => t.name === 'head.pre_classifier.weight'));
  }
  const gen = planNames(d.spec, d.head, distilNames('distilbert.', false));
  const pos = gen.tensors.find((t) => t.name === 'embeddings.position.weight')!;
  assert.equal(pos.generate, 'sinusoidal');
  assert.deepEqual(pos.shape, [6, 4]);
  // without the flag a missing position table is an error
  const learned = specFromHfConfig({ ...tinyDistil, sinusoidal_pos_embds: false }, { task: 'sequence-classification' });
  assert.throws(() => planNames(learned.spec, learned.head, distilNames('distilbert.', false)),
    /lacks required tensor distilbert\.embeddings\.position_embeddings\.weight/);
});

test('sinusoidal table: float64 angle, sin on even and cos on odd columns, one float32 rounding', () => {
  const t = sinusoidalTable(3, 4);
  const want = (pos: number, j: number) => {
    const a = pos / 10000 ** ((2 * Math.floor(j / 2)) / 4);
    return Math.fround(j % 2 === 0 ? Math.sin(a) : Math.cos(a));
  };
  assert.deepEqual([...t.subarray(0, 4)], [0, 1, 0, 1]);
  for (let pos = 0; pos < 3; pos += 1) for (let j = 0; j < 4; j += 1) assert.equal(t[pos * 4 + j], want(pos, j));
  assert.equal(t[4], Math.fround(Math.sin(1)));
});

test('names: Dense module tensors come from their own file', () => {
  const st = { modules: [], pooling: { pooling_mode_mean_tokens: true }, normalize: false,
    dense: [{ in_features: 4, out_features: 3, bias: true, activation_function: 'torch.nn.modules.activation.Tanh' }] };
  const d = specFromHfConfig(tinyDistil, { task: 'embeddings', sentenceTransformers: st });
  const plan = planNames(d.spec, d.head, distilNames('', true).filter((n) => !/classifier/.test(n)),
    { dense0: ['linear.weight', 'linear.bias'] });
  const dense = plan.tensors.filter((t) => t.file === 'dense0');
  assert.deepEqual(dense.map((t) => [t.name, t.shape]), [['head.dense0.weight', [3, 4]], ['head.dense0.bias', [3]]]);
  assert.throws(() => planNames(d.spec, d.head, distilNames('', true).filter((n) => !/classifier/.test(n)),
    { dense0: ['linear.weight'] }), /dense0 lacks required tensor linear\.bias/);
  // gather reads the Dense tensors from the extra file
  const file = parseSafetensors(safetensors({
    'linear.weight': { dtype: 'F32', shape: [3, 4], data: bytes(new Float32Array(12).fill(2)) },
    'linear.bias': { dtype: 'F32', shape: [3], data: bytes(new Float32Array(3).fill(3)) },
  }));
  const main = parseSafetensors(safetensors(Object.fromEntries(
    distilNames('', true).filter((n) => !/classifier/.test(n)).map((n) => {
      const shape = n.includes('word_embeddings') ? [5, 4] : n.includes('position') ? [6, 4]
        : n.includes('lin1.weight') ? [8, 4] : n.includes('lin1.bias') ? [8]
          : n.includes('lin2.weight') ? [4, 8]
            : /LayerNorm|layer_norm/.test(n) ? [4] : n.endsWith('weight') ? [4, 4] : [4];
      return [n, { dtype: 'F32', shape, data: bytes(new Float32Array(shape.reduce((a, b) => a * b, 1))) }];
    }))));
  const got = gatherTensors(main, plan, { dense0: file });
  assert.equal(got.find((t) => t.name === 'head.dense0.bias')!.data[0], 3);
  assert.throws(() => gatherTensors(main, plan), /additional file dense0/);
});

// K28.6: DeBERTa position projections and the ModernBERT name table.
function fakeCheckpoint(shapes: Record<string, number[]>): { names: string[]; file: ReturnType<typeof parseSafetensors> } {
  let seed = 7;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
  const tensors: Record<string, { dtype: string; shape: number[]; data: Uint8Array }> = {};
  for (const [name, shape] of Object.entries(shapes)) {
    const n = shape.reduce((a, b) => a * b, 1);
    tensors[name] = { dtype: 'F32', shape, data: bytes(Float32Array.from({ length: n }, rnd)) };
  }
  return { names: Object.keys(shapes), file: parseSafetensors(safetensors(tensors)) };
}

test('DeBERTa: pos_key and pos_query are the projections of LayerNorm(rel_embeddings), float64', () => {
  const cfg = { model_type: 'deberta-v2', hidden_size: 4, num_attention_heads: 2, num_hidden_layers: 1,
    intermediate_size: 8, vocab_size: 5, max_position_embeddings: 6, type_vocab_size: 0,
    relative_attention: true, share_att_key: true, norm_rel_ebd: 'layer_norm', pos_att_type: ['p2c', 'c2p'],
    position_biased_input: false, position_buckets: 2, layer_norm_eps: 1e-7, pooler_hidden_size: 4, id2label: { 0: 'a', 1: 'b' } };
  const { spec, head } = specFromHfConfig(cfg, { task: 'sequence-classification' });
  const p = 'deberta.encoder.layer.0.';
  const shapes: Record<string, number[]> = {
    'deberta.embeddings.word_embeddings.weight': [5, 4], 'deberta.embeddings.LayerNorm.weight': [4],
    'deberta.embeddings.LayerNorm.bias': [4], 'deberta.encoder.rel_embeddings.weight': [4, 4],
    'deberta.encoder.LayerNorm.weight': [4], 'deberta.encoder.LayerNorm.bias': [4],
    [`${p}attention.output.dense.weight`]: [4, 4], [`${p}attention.output.dense.bias`]: [4],
    [`${p}attention.output.LayerNorm.weight`]: [4], [`${p}attention.output.LayerNorm.bias`]: [4],
    [`${p}intermediate.dense.weight`]: [8, 4], [`${p}intermediate.dense.bias`]: [8],
    [`${p}output.dense.weight`]: [4, 8], [`${p}output.dense.bias`]: [4],
    [`${p}output.LayerNorm.weight`]: [4], [`${p}output.LayerNorm.bias`]: [4],
    'pooler.dense.weight': [4, 4], 'pooler.dense.bias': [4], 'classifier.weight': [2, 4], 'classifier.bias': [2],
  };
  for (const k of ['query_proj', 'key_proj', 'value_proj']) {
    shapes[`${p}attention.self.${k}.weight`] = [4, 4];
    shapes[`${p}attention.self.${k}.bias`] = [4];
  }
  const { names, file } = fakeCheckpoint(shapes);
  const plan = planNames(spec, head, names);
  assert.equal(plan.prefix, 'deberta.');
  assert.deepEqual(plan.unused, []);
  const got = new Map(gatherTensors(file, plan).map((t) => [t.name, t]));
  const rel = readF32(file, 'deberta.encoder.rel_embeddings.weight');
  const g = readF32(file, 'deberta.encoder.LayerNorm.weight');
  const b = readF32(file, 'deberta.encoder.LayerNorm.bias');
  for (const [name, proj] of [['pos_key', 'key_proj'], ['pos_query', 'query_proj']]) {
    const w = readF32(file, `${p}attention.self.${proj}.weight`);
    const bias = readF32(file, `${p}attention.self.${proj}.bias`);
    const out = got.get(`layers.0.${name}`)!;
    assert.deepEqual(out.shape, [4, 4]);
    for (let r = 0; r < 4; r += 1) {
      const row = Array.from(rel.subarray(r * 4, r * 4 + 4), Number);
      const mean = row.reduce((a, v) => a + v, 0) / 4;
      const variance = row.reduce((a, v) => a + (v - mean) ** 2, 0) / 4;
      const ln = row.map((v, c) => (v - mean) / Math.sqrt(variance + 1e-7) * g[c] + b[c]);
      for (let o = 0; o < 4; o += 1) {
        const want = ln.reduce((a, v, c) => a + v * w[o * 4 + c], bias[o]);
        assert.equal(out.data[r * 4 + o], Math.fround(want), `${name}[${r}][${o}]`);
      }
    }
  }
  assert.ok(!plan.tensors.some((t) => t.name.startsWith('embeddings.position')));
  assert.deepEqual(plan.tensors.slice(-4).map((t) => t.name),
    ['head.pooler.weight', 'head.pooler.bias', 'head.classifier.weight', 'head.classifier.bias']);
});

test('ModernBERT: names without attn_norm in layer 0 and biases only where the config has them', () => {
  const cfg = { model_type: 'modernbert', hidden_size: 4, num_attention_heads: 2, num_hidden_layers: 2,
    intermediate_size: 6, vocab_size: 5, norm_eps: 1e-5, classifier_pooling: 'mean',
    id2label: { 0: 'a', 1: 'b' } };
  const { spec, head } = specFromHfConfig(cfg, { task: 'sequence-classification' });
  const shapes: Record<string, number[]> = {
    'model.embeddings.tok_embeddings.weight': [5, 4], 'model.embeddings.norm.weight': [4],
    'model.final_norm.weight': [4], 'head.dense.weight': [4, 4], 'head.norm.weight': [4],
    'classifier.weight': [2, 4], 'classifier.bias': [2], 'decoder.bias': [5],
  };
  for (let l = 0; l < 2; l += 1) {
    const q = `model.layers.${l}.`;
    Object.assign(shapes, { [`${q}attn.Wqkv.weight`]: [12, 4], [`${q}attn.Wo.weight`]: [4, 4],
      [`${q}mlp.Wi.weight`]: [12, 4], [`${q}mlp.Wo.weight`]: [4, 6], [`${q}mlp_norm.weight`]: [4] });
    if (l > 0) shapes[`${q}attn_norm.weight`] = [4];
  }
  const { names } = fakeCheckpoint(shapes);
  const plan = planNames(spec, head, names);
  assert.deepEqual(plan.unused, ['decoder.bias']);
  const canon = plan.tensors.map((t) => t.name);
  assert.ok(!canon.includes('layers.0.attn_norm.weight') && canon.includes('layers.1.attn_norm.weight'));
  assert.ok(!canon.some((n) => n.endsWith('.bias') && !n.startsWith('head.classifier')));
  assert.deepEqual(canon.slice(-4), ['head.dense.weight', 'head.norm.weight', 'head.classifier.weight', 'head.classifier.bias']);
  // a missing required tensor and a stray tensor are errors that name the tensor
  assert.throws(() => planNames(spec, head, names.filter((n) => n !== 'head.norm.weight')), /head\.norm\.weight/);
  assert.throws(() => planNames(spec, head, [...names, 'model.layers.0.attn.rotary.inv_freq']), /rotary\.inv_freq/);
  // biases in the config require the tensors
  const biased = specFromHfConfig({ ...cfg, attention_bias: true }, { task: 'sequence-classification' });
  assert.throws(() => planNames(biased.spec, biased.head, names), /layers\.0\.attn\.Wqkv\.bias/);
});
