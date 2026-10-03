// K28.R group F: converter and tokenizer (R17, R30, R18, R25, T01, R07, R11). R17, R30, R18, R25
// and R11 are the review tests, unchanged. T01 and R07 are new contracts: the ordered fix is a
// rejection at load, the review contracts (Unicode word boundary; type ids bound in the plan)
// stay red in tests/review. The last test pins all frozen configs of the list to a spec.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { specFromHfConfig } from '../src/plan/hf.ts';
import { gatherTensors } from '../src/convert/manifest.ts';
import { parseSafetensors, readF32 } from '../src/convert/safetensors.ts';
import { planNames } from '../src/convert/names.ts';
import { Bpe } from '../src/tokenizer/hf/bpe.ts';
import { oniguruma, parseNormalizer } from '../src/tokenizer/hf/normalizers.ts';
import { fromText, join } from '../src/tokenizer/hf/normalized.ts';
import { DISTILUSE_DENSE } from './helpers/pilot.ts';

function safetensors(header, data) {
  const h = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(8 + h.length + data.length);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(h.length), true);
  bytes.set(h, 8);
  bytes.set(data, 8 + h.length);
  return bytes;
}
const f32 = (...v) => new Uint8Array(Float32Array.of(...v).buffer);

// ---------- R17 ----------

test('R17 safetensors must reject truncated data instead of zero filling', () => {
  const header = new TextEncoder().encode(JSON.stringify({ w: { dtype: 'F32', shape: [2], data_offsets: [0, 8] } }));
  const bytes = new Uint8Array(8 + header.length + 4);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(header.length), true);
  bytes.set(header, 8);
  new DataView(bytes.buffer).setFloat32(8 + header.length, 42, true);
  assert.throws(() => readF32(parseSafetensors(bytes), 'w'), /offset|size|trunc|length/i);
});

test('R17 header checks: dtype, shape, byte count, offsets, bounds, overlap; a valid file parses', () => {
  const ok = { a: { dtype: 'F32', shape: [2], data_offsets: [0, 8] }, b: { dtype: 'F16', shape: [2, 2], data_offsets: [8, 16] } };
  const file = parseSafetensors(safetensors({ __metadata__: { format: 'pt' }, ...ok }, new Uint8Array(16)));
  assert.deepEqual([...file.tensors.keys()], ['a', 'b']);
  const bad = (header, bytes, re) => assert.throws(
    () => parseSafetensors(safetensors(header, new Uint8Array(bytes))), re, JSON.stringify(header));
  bad({ w: { dtype: 'F33', shape: [1], data_offsets: [0, 4] } }, 4, /w has unknown dtype F33/);
  bad({ w: { dtype: 'F32', shape: [-1], data_offsets: [0, 4] } }, 4, /w has shape/);
  bad({ w: { dtype: 'F32', shape: [1.5], data_offsets: [0, 4] } }, 4, /w has shape/);
  bad({ w: { dtype: 'F32', shape: [2], data_offsets: [0, 4] } }, 8, /w has 4 bytes, F32 2 needs 8/);
  bad({ w: { dtype: 'F32', shape: [1], data_offsets: [4, 0] } }, 8, /w has data_offsets/);
  bad({ w: { dtype: 'F32', shape: [1], data_offsets: [-4, 0] } }, 8, /w has data_offsets/);
  bad({ w: { dtype: 'F32', shape: [1], data_offsets: [0, 4.5] } }, 8, /w has data_offsets/);
  bad({ w: { dtype: 'F32', shape: [1], data_offsets: [0, 4] } }, 2, /w has data_offsets/);
  bad({ a: { dtype: 'F32', shape: [2], data_offsets: [0, 8] }, b: { dtype: 'F32', shape: [2], data_offsets: [4, 12] } },
    12, /b overlaps a/);
  // two tensors may share an empty range
  assert.doesNotThrow(() => parseSafetensors(safetensors({
    a: { dtype: 'F32', shape: [1], data_offsets: [0, 4] }, e: { dtype: 'F32', shape: [0], data_offsets: [4, 4] } },
  new Uint8Array(4))));
});

// ---------- R30 ----------

test('R30 Conv1d shape validation must distinguish input width from kernel width', () => {
  const data = Float32Array.from({ length: 27 }, (_, i) => i);
  const file = { bytes: new Uint8Array(data.buffer), tensors: new Map([
    ['w', { name: 'w', dtype: 'F32', shape: [3, 1, 9], start: 0, end: data.byteLength }],
  ]) };
  // Plan from a one-group H=3, kernel=3 conv expects source [3,3,3].
  // [3,1,9] has the same element count, but its layout is NOT equivalent.
  const names = { prefix: '', unused: [], tensors: [
    { name: 'conv.weight', sources: ['w'], shape: [3, 9], transform: 'conv1d' },
  ] };
  assert.throws(() => gatherTensors(file, names), /shape|conv|dimension/i);
});

test('R30 a conv source of [H, H, k] is gathered, [H, k, H] and a flat source are not', () => {
  const data = Float32Array.from({ length: 27 }, (_, i) => i);
  const names = { prefix: '', unused: [], tensors: [
    { name: 'conv.weight', sources: ['w'], shape: [3, 9], transform: 'conv1d' }] };
  const of = (shape) => ({ bytes: new Uint8Array(data.buffer), tensors: new Map([
    ['w', { name: 'w', dtype: 'F32', shape, start: 0, end: data.byteLength }]]) });
  const [t] = gatherTensors(of([3, 3, 3]), names);
  assert.deepEqual([...t.data.slice(0, 9)], [0, 3, 6, 1, 4, 7, 2, 5, 8]);
  assert.throws(() => gatherTensors(of([3, 9]), names), /expected 3x3x3/);
  assert.throws(() => gatherTensors(of([9, 3, 1]), names), /expected 3x3x3/);
});

// ---------- R18 ----------

test('R18 HF BPE pair IDs must not collide above 2^20', () => {
  const bpe = new Bpe(new Map([['a', 0], ['x', 1], ['xx', 2], ['y', 1048577], ['ay', 3]]),
    [['a', 'y'], ['x', 'x']], { unkToken: null, continuingSubwordPrefix: null,
      endOfWordSuffix: null, fuseUnk: false, byteFallback: false, ignoreMerges: false });
  assert.equal(bpe.tokenize(['a', 'y'])[0].id, 3);
});

test('R18 an id too large for an exact pair key is rejected at load', () => {
  const opt = { unkToken: null, continuingSubwordPrefix: null, endOfWordSuffix: null,
    fuseUnk: false, byteFallback: false, ignoreMerges: false };
  assert.throws(() => new Bpe(new Map([['a', 0], ['b', 2 ** 27]]), [], opt), /too large/);
  assert.throws(() => new Bpe(new Map([['a', -1]]), [], opt), /non-negative integer/);
  assert.doesNotThrow(() => new Bpe(new Map([['a', 0], ['b', 2 ** 26]]), [], opt));
});

// ---------- R25 ----------

test('R25 bias-free Sentence-Transformers Dense must not require linear.bias', () => {
  const { spec, head } = specFromHfConfig({
    model_type: 'bert', hidden_size: 4, num_hidden_layers: 0, num_attention_heads: 1,
    intermediate_size: 8, vocab_size: 2, type_vocab_size: 0,
  }, { task: 'embeddings', sentenceTransformers: {
    pooling: { pooling_mode_mean_tokens: true },
    dense: [{ in_features: 4, out_features: 3, bias: false, activation_function: 'torch.nn.modules.linear.Identity' }],
  } });
  assert.doesNotThrow(() => planNames(spec, head, [
    'embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight',
    'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias',
  ], { dense0: ['linear.weight'] }));
});

// ---------- T01 ----------

test('T01 \\b and \\B outside a character class are rejected at load; inside a class they stay', () => {
  for (const pattern of ['\\bfoo\\b', 'a\\Bb', 'x(\\b)', '[a]\\b']) {
    assert.throws(() => parseNormalizer({ type: 'Replace', pattern: { Regex: pattern }, content: 'X' }),
      /unsupported \\[bB] outside a character class/, pattern);
  }
  // [\b] is a backspace in both engines; an escaped backslash followed by b is no boundary
  assert.doesNotThrow(() => oniguruma('[\\b]x'));
  assert.doesNotThrow(() => oniguruma('\\\\bx'));
  const normalize = parseNormalizer({ type: 'Replace', pattern: { Regex: '\\s+' }, content: ' ' });
  assert.equal(join(normalize(fromText('a  \n b'))), 'a b');
});

// ---------- R07, R11 ----------

const DEBERTA = {
  model_type: 'deberta-v2', hidden_size: 64, num_hidden_layers: 1, num_attention_heads: 1,
  intermediate_size: 128, vocab_size: 8, max_position_embeddings: 512,
  relative_attention: true, share_att_key: true, norm_rel_ebd: 'layer_norm',
  position_buckets: 256, pos_att_type: ['p2c', 'c2p'],
};

test('R07 DeBERTa with a type table and no absolute positions is rejected at load', () => {
  assert.throws(() => specFromHfConfig({ ...DEBERTA, position_biased_input: false, type_vocab_size: 2 },
    { task: 'token-classification' }), /type_vocab_size/);
  // the combinations that stay: no type table, or absolute positions with a type table
  assert.doesNotThrow(() => specFromHfConfig({ ...DEBERTA, position_biased_input: false, type_vocab_size: 0 },
    { task: 'token-classification' }));
  assert.doesNotThrow(() => specFromHfConfig({ ...DEBERTA, position_biased_input: true, type_vocab_size: 2 },
    { task: 'token-classification' }));
});

test('R11 BERT decoder config must be rejected instead of full noncausal attention', () => {
  assert.throws(() => specFromHfConfig({
    model_type: 'bert', hidden_size: 64, num_hidden_layers: 1, num_attention_heads: 1,
    intermediate_size: 128, vocab_size: 8, is_decoder: true,
  }, { task: 'token-classification' }), /is_decoder|causal/);
});

test('R11 is_decoder and add_cross_attention are rejected for bert, electra, roberta and xlm-roberta', () => {
  for (const model_type of ['bert', 'electra', 'roberta', 'xlm-roberta']) {
    const base = { model_type, hidden_size: 64, num_hidden_layers: 1, num_attention_heads: 1,
      intermediate_size: 128, vocab_size: 8, max_position_embeddings: 512 };
    const extras = { task: 'token-classification', template: [0, null, 2] };
    assert.doesNotThrow(() => specFromHfConfig({ ...base, is_decoder: false, add_cross_attention: false }, extras), model_type);
    assert.throws(() => specFromHfConfig({ ...base, is_decoder: true }, extras), /is_decoder/, model_type);
    assert.throws(() => specFromHfConfig({ ...base, add_cross_attention: true }, extras), /add_cross_attention/, model_type);
  }
});

// ---------- all frozen configs ----------

test('all frozen configs of data/k28/models.json still produce a spec', () => {
  const models = JSON.parse(readFileSync('data/k28/models.json', 'utf8')).models;
  assert.equal(models.length, 89);
  for (const e of models) {
    const st = e.sentenceTransformers;
    const sentenceTransformers = st && st.dense === true ? { ...st, dense: DISTILUSE_DENSE } : st;
    const { spec, head } = specFromHfConfig(e.config, { task: e.task, sentenceTransformers,
      template: [e.config.bos_token_id, null, e.config.eos_token_id] });
    assert.ok(spec.hidden > 0 && head.type, e.id);
  }
});
