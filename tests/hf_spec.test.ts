// K28.2 step 3, K28.5 step 2, K28.6 step 3: ModelSpec and HeadSpec from the config.json of the
// eight BERT pilot models, the sixteen RoBERTa, XLM-R and DistilBERT models and the eleven DeBERTa
// and ModernBERT models (data/k28/models.json, frozen revisions), plus one error case
// per rejected key. Run with the Node hooks (npm test).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { manifestMaxLength, specFromHfConfig, templateOfTokenizer, type HfExtras, type Task } from '../src/plan/hf.ts';
import { DISTILUSE_DENSE } from './helpers/pilot.ts';

interface Entry {
  id: string; task: Task; config: Record<string, unknown>;
  sentenceTransformers?: HfExtras['sentenceTransformers'];
}

const models = (JSON.parse(readFileSync(
  new URL('../data/k28/models.json', import.meta.url), 'utf8')) as { models: Entry[] }).models;
const entry = (id: string): Entry => {
  const e = models.find((m) => m.id === id);
  assert.ok(e, id);
  return e;
};
// <s> A </s>, the single template of the RoBERTa and XLM-R tokenizers of the list
const templateOf = (e: Entry): Array<number | null> =>
  [e.config.bos_token_id as number, null, e.config.eos_token_id as number];

const build = (e: Entry, patch: Record<string, unknown> = {}, task: Task = e.task) =>
  specFromHfConfig({ ...e.config, ...patch },
    { task, sentenceTransformers: e.sentenceTransformers, template: templateOf(e) });

const dense = (name: string, i: number, o: number, act: string) =>
  ({ op: 'dense', name, in: i, out: o, act, bias: true });
const pooled = (h: number, c: number) => [dense('head.pooler', h, h, 'tanh'), dense('head.classifier', h, c, 'none')];

const EXPECT: Record<string, {
  hidden: number; layers: number; heads: number; headDim: number; intermediate: number;
  embeddingSize: number; vocab: number; family: string; project: boolean; head: object;
}> = {
  'daekeun-ml/koelectra-small-v3-nsmc': {
    hidden: 256, layers: 12, heads: 4, headDim: 64, intermediate: 1024, embeddingSize: 128,
    vocab: 35000, family: 'electra', project: true,
    head: { type: 'classify', pool: 'first', classes: 2,
      steps: [dense('head.dense', 256, 256, 'gelu'), dense('head.out_proj', 256, 2, 'none')] },
  },
  'MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33': {
    hidden: 256, layers: 6, heads: 8, headDim: 32, intermediate: 1024, embeddingSize: 256,
    vocab: 30522, family: 'bert', project: false,
    head: { type: 'classify', pool: 'first', classes: 2, steps: pooled(256, 2) },
  },
  'sentence-transformers/all-MiniLM-L6-v2': {
    hidden: 384, layers: 6, heads: 12, headDim: 32, intermediate: 1536, embeddingSize: 384,
    vocab: 30522, family: 'bert', project: false,
    head: { type: 'embed', pool: 'mean', steps: [], normalize: true },
  },
  'BAAI/bge-small-en-v1.5': {
    hidden: 384, layers: 12, heads: 12, headDim: 32, intermediate: 1536, embeddingSize: 384,
    vocab: 30522, family: 'bert', project: false,
    head: { type: 'embed', pool: 'cls', steps: [], normalize: true },
  },
  'cross-encoder/ms-marco-MiniLM-L6-v2': {
    hidden: 384, layers: 6, heads: 12, headDim: 32, intermediate: 1536, embeddingSize: 384,
    vocab: 30522, family: 'bert', project: false,
    head: { type: 'classify', pool: 'first', classes: 1, steps: pooled(384, 1) },
  },
  'cross-encoder/ms-marco-MiniLM-L4-v2': {
    hidden: 384, layers: 4, heads: 12, headDim: 32, intermediate: 1536, embeddingSize: 384,
    vocab: 30522, family: 'bert', project: false,
    head: { type: 'classify', pool: 'first', classes: 1, steps: pooled(384, 1) },
  },
  'cross-encoder/ms-marco-MiniLM-L12-v2': {
    hidden: 384, layers: 12, heads: 12, headDim: 32, intermediate: 1536, embeddingSize: 384,
    vocab: 30522, family: 'bert', project: false,
    head: { type: 'classify', pool: 'first', classes: 1, steps: pooled(384, 1) },
  },
  'dslim/bert-base-NER': {
    hidden: 768, layers: 12, heads: 12, headDim: 64, intermediate: 3072, embeddingSize: 768,
    vocab: 28996, family: 'bert', project: false,
    head: { type: 'token', classes: 9, steps: [dense('head.classifier', 768, 9, 'none')] },
  },
};

for (const [id, want] of Object.entries(EXPECT)) {
  test(`spec of ${id}`, () => {
    const { spec, head } = build(entry(id));
    assert.equal(spec.family, want.family);
    for (const k of ['hidden', 'layers', 'heads', 'headDim', 'intermediate', 'embeddingSize', 'vocab'] as const) {
      assert.equal(spec[k], want[k], k);
    }
    assert.equal(spec.embed.typeVocab, 2);
    assert.equal(spec.embed.maxPositions, 512);
    assert.equal(spec.embed.positions, 'absolute');
    assert.equal(spec.embed.positionOffset, 0);
    assert.equal(spec.embed.project, want.project);
    assert.equal(spec.embed.maskMultiply, false);
    assert.deepEqual(spec.embed.norm, { eps: 1e-12, bias: true });
    assert.deepEqual(spec.block, { order: 'post', firstNormIdentity: false, finalNorm: false,
      norm: { eps: 1e-12, bias: true } });
    assert.deepEqual(spec.attention, { kind: 'standard', bias: true, scale: want.headDim ** -0.5 });
    assert.deepEqual(spec.ffn, { kind: 'mlp', act: 'gelu', bias: true });
    assert.deepEqual(head, want.head);
  });
}

test('Dense modules become head steps', () => {
  const e = entry('sentence-transformers/all-MiniLM-L6-v2');
  const st = { ...e.sentenceTransformers, dense: [{ in_features: 384, out_features: 512, bias: true,
    activation_function: 'torch.nn.modules.activation.Tanh' }] };
  const { head } = specFromHfConfig(e.config, { task: 'embeddings', sentenceTransformers: st });
  assert.deepEqual(head, { type: 'embed', pool: 'mean', normalize: true,
    steps: [{ op: 'dense', name: 'head.dense0', in: 384, out: 512, act: 'tanh', bias: true }] });
});

// One error per rejected key; the message names the key.
const ERRORS: [string, string, Record<string, unknown>, RegExp, Task?][] = [
  ['other family', 'dslim/bert-base-NER', { model_type: 'albert' }, /'model_type'/],
  ['position_embedding_type', 'dslim/bert-base-NER', { position_embedding_type: 'relative_key' }, /'position_embedding_type'/],
  ['hidden_act', 'dslim/bert-base-NER', { hidden_act: 'gelu_new' }, /'hidden_act'/],
  ['head dimension', 'dslim/bert-base-NER', { num_attention_heads: 8 }, /head dimension 96 exceeds 64/],
  ['heads not dividing', 'dslim/bert-base-NER', { num_attention_heads: 7 }, /'num_attention_heads'/],
  ['reranking classes', 'cross-encoder/ms-marco-MiniLM-L6-v2',
    { id2label: { 0: 'a', 1: 'b' } }, /'id2label'/],
  ['NLI without entailment', 'MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33',
    { id2label: { 0: 'yes', 1: 'no' } }, /'id2label'/],
  ['missing key', 'dslim/bert-base-NER', { hidden_size: undefined }, /'hidden_size'/],
];
for (const [name, id, patch, re] of ERRORS) {
  test(`rejects ${name}`, () => assert.throws(() => build(entry(id), patch), re));
}

test('rejects unsupported pooling', () => {
  const e = entry('sentence-transformers/all-MiniLM-L6-v2');
  const st = { ...e.sentenceTransformers, pooling: { pooling_mode_lasttoken: true } };
  assert.throws(() => specFromHfConfig(e.config, { task: 'embeddings', sentenceTransformers: st }),
    /'pooling_mode_lasttoken'/);
});
test('rejects two pooling modes', () => {
  const e = entry('sentence-transformers/all-MiniLM-L6-v2');
  const st = { ...e.sentenceTransformers,
    pooling: { pooling_mode_cls_token: true, pooling_mode_mean_tokens: true } };
  assert.throws(() => specFromHfConfig(e.config, { task: 'embeddings', sentenceTransformers: st }),
    /'pooling_mode'/);
});
test('rejects unknown Dense activation', () => {
  const e = entry('sentence-transformers/all-MiniLM-L6-v2');
  const st = { ...e.sentenceTransformers, dense: [{ in_features: 384, out_features: 8, bias: true,
    activation_function: 'torch.nn.modules.activation.Softplus' }] };
  assert.throws(() => specFromHfConfig(e.config, { task: 'embeddings', sentenceTransformers: st }),
    /'activation_function'/);
});

// K28.5: the sixteen models of RoBERTa, XLM-R and DistilBERT.
const step = (name: string, i: number, o: number, act: string) => dense(name, i, o, act);
const roberta = (h: number, c: number) => [step('head.dense', h, h, 'tanh'), step('head.out_proj', h, c, 'none')];
const distil = (c: number) => [step('head.pre_classifier', 768, 768, 'relu'), step('head.classifier', 768, c, 'none')];
const seq = (steps: object[], classes: number) => ({ type: 'classify', pool: 'first', classes, steps });

const EXPECT285: Record<string, {
  family: string; hidden: number; layers: number; vocab: number; intermediate: number;
  maxPositions: number; offset: number; typeVocab: number; eps: number; head: object;
  sinusoidal?: boolean; maxLength?: number;
}> = {
  'cardiffnlp/twitter-roberta-base-sentiment-latest': { family: 'roberta', hidden: 768, layers: 12,
    vocab: 50265, intermediate: 3072, maxPositions: 514, offset: 2, typeVocab: 1, eps: 1e-5,
    head: seq(roberta(768, 3), 3) },
  'cross-encoder/nli-distilroberta-base': { family: 'roberta', hidden: 768, layers: 6,
    vocab: 50265, intermediate: 3072, maxPositions: 514, offset: 2, typeVocab: 1, eps: 1e-5,
    head: seq(roberta(768, 3), 3) },
  'sentence-transformers/all-distilroberta-v1': { family: 'roberta', hidden: 768, layers: 6,
    vocab: 50265, intermediate: 3072, maxPositions: 514, offset: 2, typeVocab: 1, eps: 1e-5,
    head: { type: 'embed', pool: 'mean', steps: [], normalize: true } },
  'OpenMed/OpenMed-NER-OrganismDetect-TinyMed-82M': { family: 'roberta', hidden: 768, layers: 6,
    vocab: 50265, intermediate: 3072, maxPositions: 514, offset: 2, typeVocab: 1, eps: 1e-7,
    head: { type: 'token', classes: 3, steps: [step('head.classifier', 768, 3, 'none')] } },
  'cross-encoder/stsb-distilroberta-base': { family: 'roberta', hidden: 768, layers: 6,
    vocab: 50265, intermediate: 3072, maxPositions: 514, offset: 2, typeVocab: 1, eps: 1e-5,
    head: seq(roberta(768, 1), 1) },
  'qilowoq/mmarco-mMiniLMv2-L12-H384-v1-en-ru': { family: 'xlm-roberta', hidden: 384, layers: 12,
    vocab: 60352, intermediate: 1536, maxPositions: 514, offset: 2, typeVocab: 1, eps: 1e-5,
    head: seq(roberta(384, 1), 1) },
  'MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli': { family: 'xlm-roberta', hidden: 384, layers: 6,
    vocab: 250002, intermediate: 1536, maxPositions: 514, offset: 2, typeVocab: 1, eps: 1e-5,
    head: seq(roberta(384, 3), 3) },
  // <s> and <pad> are both id 0: <s> counts as padding, positions start at row 0.
  'd0rj/e5-small-en-ru': { family: 'xlm-roberta', hidden: 384, layers: 12,
    vocab: 60302, intermediate: 1536, maxPositions: 512, offset: 0, typeVocab: 2, eps: 1e-12,
    head: { type: 'embed', pool: 'mean', steps: [], normalize: true } },
  'ukr-models/uk-ner': { family: 'xlm-roberta', hidden: 768, layers: 12,
    vocab: 31274, intermediate: 3072, maxPositions: 514, offset: 2, typeVocab: 1, eps: 1e-5,
    head: { type: 'token', classes: 7, steps: [step('head.classifier', 768, 7, 'none')] } },
  'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1': { family: 'xlm-roberta', hidden: 384, layers: 12,
    vocab: 250002, intermediate: 1536, maxPositions: 514, offset: 2, typeVocab: 1, eps: 1e-5,
    head: seq(roberta(384, 1), 1) },
  'distilbert/distilbert-base-uncased-finetuned-sst-2-english': { family: 'distilbert', hidden: 768,
    layers: 6, vocab: 30522, intermediate: 3072, maxPositions: 512, offset: 0, typeVocab: 0,
    eps: 1e-12, head: seq(distil(2), 2) },
  'typeform/distilbert-base-uncased-mnli': { family: 'distilbert', hidden: 768,
    layers: 6, vocab: 30522, intermediate: 3072, maxPositions: 512, offset: 0, typeVocab: 0,
    eps: 1e-12, head: seq(distil(3), 3) },
  'sentence-transformers/distiluse-base-multilingual-cased-v1': { family: 'distilbert', hidden: 768,
    layers: 6, vocab: 119547, intermediate: 3072, maxPositions: 512, offset: 0, typeVocab: 0,
    eps: 1e-12, maxLength: 128,
    head: { type: 'embed', pool: 'mean', normalize: false,
      steps: [step('head.dense0', 768, 512, 'tanh')] } },
  'OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-65M': { family: 'distilbert', hidden: 768,
    layers: 6, vocab: 28996, intermediate: 3072, maxPositions: 512, offset: 0, typeVocab: 0,
    eps: 1e-12, head: { type: 'token', classes: 3, steps: [step('head.classifier', 768, 3, 'none')] } },
  'Amdestya/ce-cat-distilbert': { family: 'distilbert', hidden: 768,
    layers: 6, vocab: 30522, intermediate: 3072, maxPositions: 512, offset: 0, typeVocab: 0,
    eps: 1e-12, head: seq(distil(1), 1) },
  'emrecan/distilbert-base-turkish-cased-allnli_tr': { family: 'distilbert', hidden: 768,
    layers: 6, vocab: 32000, intermediate: 3072, maxPositions: 512, offset: 0, typeVocab: 0,
    eps: 1e-12, sinusoidal: true, head: seq(distil(3), 3) },
};

const withDense = (e: Entry): HfExtras['sentenceTransformers'] => {
  const st = e.sentenceTransformers;
  return st && st.dense === true ? { ...st, dense: DISTILUSE_DENSE } : st;
};

for (const [id, want] of Object.entries(EXPECT285)) {
  test(`spec of ${id}`, () => {
    const e = entry(id);
    const { spec, head } = specFromHfConfig(e.config, {
      task: e.task, sentenceTransformers: withDense(e), template: templateOf(e) });
    assert.equal(spec.family, want.family);
    assert.equal(spec.hidden, want.hidden);
    assert.equal(spec.layers, want.layers);
    assert.equal(spec.heads, 12);
    assert.equal(spec.headDim, want.hidden / 12);
    assert.equal(spec.intermediate, want.intermediate);
    assert.equal(spec.vocab, want.vocab);
    assert.equal(spec.embeddingSize, want.hidden);
    assert.equal(spec.embed.maxPositions, want.maxPositions);
    assert.equal(spec.embed.positionOffset, want.offset);
    assert.equal(spec.embed.typeVocab, want.typeVocab);
    assert.equal(spec.embed.project, false);
    assert.equal(spec.embed.sinusoidal, want.sinusoidal);
    assert.deepEqual(spec.embed.norm, { eps: want.eps, bias: true });
    assert.deepEqual(spec.block.norm, { eps: want.eps, bias: true });
    assert.deepEqual(spec.ffn, { kind: 'mlp', act: 'gelu', bias: true });
    assert.deepEqual(head, want.head);
    // maxLength: position rows after the offset, tokenizer cap, max_seq_length
    const st = withDense(e);
    assert.equal(manifestMaxLength(spec, 512, st?.maxSeqLength),
      want.maxLength ?? Math.min(want.maxPositions - want.offset, 512, st?.maxSeqLength ?? Infinity));
  });
}

test('manifestMaxLength ignores the unlimited model_max_length placeholder', () => {
  const { spec } = build(entry('dslim/bert-base-NER'));
  assert.equal(manifestMaxLength(spec, 1e30), 512);
  assert.equal(manifestMaxLength(spec, 256, 128), 128);
  assert.equal(manifestMaxLength(spec, null, null), 512);
});

const ERRORS285: [string, string, Record<string, unknown>, RegExp][] = [
  ['distilbert activation', 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    { activation: 'gelu_new' }, /'activation'/],
  ['distilbert head dimension', 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    { n_heads: 8 }, /head dimension 96 exceeds 64/],
  ['distilbert missing dim', 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    { dim: undefined }, /'dim'/],
  ['distilbert position type', 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    { position_embedding_type: 'relative_key' }, /'position_embedding_type'/],
  ['roberta hidden_act', 'cross-encoder/nli-distilroberta-base', { hidden_act: 'gelu_new' }, /'hidden_act'/],
  ['roberta position type', 'cross-encoder/nli-distilroberta-base',
    { position_embedding_type: 'relative_key_query' }, /'position_embedding_type'/],
  ['roberta position rows', 'cross-encoder/nli-distilroberta-base',
    { max_position_embeddings: 2 }, /'max_position_embeddings'/],
  ['xlm-roberta head dimension', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
    { num_attention_heads: 4 }, /head dimension 96 exceeds 64/],
  ['xlm-roberta reranking classes', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
    { id2label: { 0: 'a', 1: 'b' } }, /'id2label'/],
  ['distilbert NLI without entailment', 'typeform/distilbert-base-uncased-mnli',
    { id2label: { 0: 'yes', 1: 'no' } }, /'id2label'/],
];
for (const [name, id, patch, re] of ERRORS285) {
  test(`rejects ${name}`, () => assert.throws(() => build(entry(id), patch), re));
}

test('a Dense module without its config is an error naming the key', () => {
  const e = entry('sentence-transformers/distiluse-base-multilingual-cased-v1');
  assert.throws(() => specFromHfConfig(e.config, { task: 'embeddings', sentenceTransformers: e.sentenceTransformers }),
    /'dense'/);
});
test('a Dense module with the wrong input width is rejected', () => {
  const e = entry('sentence-transformers/distiluse-base-multilingual-cased-v1');
  const st = { ...e.sentenceTransformers, dense: [{ ...DISTILUSE_DENSE[0], in_features: 384 }] };
  assert.throws(() => specFromHfConfig(e.config, { task: 'embeddings', sentenceTransformers: st }),
    /'in_features'/);
});

test('RoBERTa position offset follows the single template of tokenizer.json, not bos_token_id', () => {
  const e = entry('cross-encoder/nli-distilroberta-base'); // pad_token_id 1
  const offset = (template: Array<number | null> | undefined, patch: Record<string, unknown> = {}) =>
    specFromHfConfig({ ...e.config, ...patch }, { task: e.task, template }).spec.embed.positionOffset;
  assert.equal(offset([0, null, 2]), 2);
  assert.equal(offset([1, null, 2]), 1); // the first token is the pad id: it counts as padding
  assert.equal(offset([null]), 2); // no leading special token: a text token starts the sequence
  assert.equal(offset([0, null, 2], { bos_token_id: 1 }), 2); // bos_token_id in the config is not read
  assert.throws(() => offset([0, null, 1]), /later special token/);
  assert.throws(() => offset(undefined), /single template/);
});

test('templateOfTokenizer reads Template, Roberta and ByteLevel post-processors', () => {
  const template = { type: 'TemplateProcessing', single: [{ SpecialToken: { id: '<s>', type_id: 0 } },
    { Sequence: { id: 'A', type_id: 0 } }, { SpecialToken: { id: '</s>', type_id: 0 } }],
  special_tokens: { '<s>': { ids: [5] }, '</s>': { ids: [6] } } };
  assert.deepEqual(templateOfTokenizer({ post_processor: template }), [5, null, 6]);
  assert.deepEqual(templateOfTokenizer({ post_processor: { type: 'RobertaProcessing', sep: ['</s>', 2], cls: ['<s>', 0] } }),
    [0, null, 2]);
  assert.deepEqual(templateOfTokenizer({ post_processor: { type: 'ByteLevel' } }), [null]);
  assert.deepEqual(templateOfTokenizer({ post_processor: null }), [null]);
  assert.throws(() => templateOfTokenizer({ post_processor: { type: 'BertProcessing' } }), /BertProcessing/);
});

// K28.6: DeBERTa-v2 and v3, ModernBERT.
const denseK = (name: string, i: number, o: number, act: string, bias = true) =>
  ({ op: 'dense', name, in: i, out: o, act, bias });
const debertaPooled = (h: number, c: number) =>
  [denseK('head.pooler', h, h, 'gelu'), denseK('head.classifier', h, c, 'none')];
const mbHead = (h: number, c: number, act: string, denseBias = false) => [
  denseK('head.dense', h, h, act, denseBias), { op: 'norm', name: 'head.norm', eps: 1e-5, bias: false },
  denseK('head.classifier', h, c, 'none')];

const EXPECT_DEBERTA: Record<string, {
  hidden: number; layers: number; heads: number; headDim: number; intermediate: number; vocab: number;
  head: object;
}> = {
  'protectai/deberta-v3-base-prompt-injection-v2': {
    hidden: 768, layers: 12, heads: 12, headDim: 64, intermediate: 3072, vocab: 128100,
    head: { type: 'classify', pool: 'first', classes: 2, steps: debertaPooled(768, 2) },
  },
  'cross-encoder/nli-deberta-v3-small': {
    hidden: 768, layers: 6, heads: 12, headDim: 64, intermediate: 3072, vocab: 128100,
    head: { type: 'classify', pool: 'first', classes: 3, steps: debertaPooled(768, 3) },
  },
  'OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-141M': {
    hidden: 768, layers: 6, heads: 12, headDim: 64, intermediate: 3072, vocab: 128100,
    head: { type: 'token', classes: 11, steps: [denseK('head.classifier', 768, 11, 'none')] },
  },
  'mixedbread-ai/mxbai-rerank-xsmall-v1': {
    hidden: 384, layers: 12, heads: 6, headDim: 64, intermediate: 1536, vocab: 128100,
    head: { type: 'classify', pool: 'first', classes: 1, steps: debertaPooled(384, 1) },
  },
};
for (const [id, want] of Object.entries(EXPECT_DEBERTA)) {
  test(`spec of ${id}`, () => {
    const { spec, head } = specFromHfConfig(entry(id).config, { task: entry(id).task });
    assert.equal(spec.family, 'deberta-v2');
    for (const k of ['hidden', 'layers', 'heads', 'headDim', 'intermediate', 'vocab'] as const) {
      assert.equal(spec[k], want[k], k);
    }
    assert.deepEqual(spec.embed, { positions: 'none', positionOffset: 0, maxPositions: 512, typeVocab: 0,
      norm: { eps: 1e-7, bias: true }, maskMultiply: true, project: false });
    assert.deepEqual(spec.attention, { kind: 'deberta-relative', bias: true, scale: 1 / Math.sqrt(3 * want.headDim),
      rel: { buckets: 256, maxPositions: 512, types: ['c2p', 'p2c'] } });
    assert.deepEqual(spec.block, { order: 'post', firstNormIdentity: false, finalNorm: false,
      norm: { eps: 1e-7, bias: true } });
    assert.deepEqual(head, want.head);
  });
}

test('spec of polyBERT: no relative attention, absolute positions, head width 50', () => {
  const e = entry('xushijie/polyBERT');
  const { spec, head } = specFromHfConfig(e.config, { task: e.task, sentenceTransformers: e.sentenceTransformers });
  assert.deepEqual([spec.hidden, spec.layers, spec.heads, spec.headDim, spec.intermediate, spec.vocab],
    [600, 12, 12, 50, 512, 269]);
  assert.deepEqual(spec.embed, { positions: 'absolute', positionOffset: 0, maxPositions: 512, typeVocab: 0,
    norm: { eps: 1e-7, bias: true }, maskMultiply: true, project: false });
  assert.deepEqual(spec.attention, { kind: 'standard', bias: true, scale: 50 ** -0.5 });
  assert.deepEqual(head, { type: 'embed', pool: 'mean', steps: [], normalize: false });
});

const EXPECT_MB: Record<string, {
  hidden: number; layers: number; heads: number; intermediate: number; vocab: number; theta: [number, number];
  half: number; head: object;
}> = {
  'sheltron-ai/prompt-guard-68m': {
    hidden: 512, layers: 19, heads: 8, intermediate: 768, vocab: 50368, theta: [160000, 160000], half: 64,
    head: { type: 'classify', pool: 'mean', classes: 22, steps: mbHead(512, 22, 'gelu') },
  },
  'Horizon-Labs/multilingual-zeroshot-small': {
    hidden: 384, layers: 22, heads: 6, intermediate: 1152, vocab: 256000, theta: [160000, 160000], half: 64,
    head: { type: 'classify', pool: 'mean', classes: 2, steps: mbHead(384, 2, 'gelu') },
  },
  'ibm-granite/granite-embedding-small-english-r2': {
    hidden: 384, layers: 12, heads: 12, intermediate: 1536, vocab: 50368, theta: [80000, 10000], half: 64,
    head: { type: 'embed', pool: 'cls', steps: [], normalize: false },
  },
  'OpenMed/OpenMed-NER-ChemicalDetect-ModernMed-149M': {
    hidden: 768, layers: 22, heads: 12, intermediate: 1152, vocab: 50368, theta: [160000, 10000], half: 64,
    head: { type: 'token', classes: 3, steps: mbHead(768, 3, 'gelu') },
  },
  'hotchpotch/japanese-reranker-xsmall-v2': {
    hidden: 256, layers: 10, heads: 4, intermediate: 1024, vocab: 102400, theta: [160000, 10000], half: 64,
    head: { type: 'classify', pool: 'first', classes: 1, steps: mbHead(256, 1, 'gelu') },
  },
  'ibm-granite/granite-embedding-reranker-english-r2': {
    hidden: 768, layers: 22, heads: 12, intermediate: 1152, vocab: 50368, theta: [160000, 40000], half: 64,
    head: { type: 'classify', pool: 'first', classes: 1, steps: mbHead(768, 1, 'silu') },
  },
};
for (const [id, want] of Object.entries(EXPECT_MB)) {
  test(`spec of ${id}`, () => {
    const e = entry(id);
    const { spec, head } = specFromHfConfig(e.config, { task: e.task, sentenceTransformers: e.sentenceTransformers });
    assert.equal(spec.family, 'modernbert');
    for (const k of ['hidden', 'layers', 'heads', 'intermediate', 'vocab'] as const) assert.equal(spec[k], want[k], k);
    assert.equal(spec.headDim, want.hidden / want.heads);
    assert.deepEqual(spec.embed, { positions: 'none', positionOffset: 0, maxPositions: e.config.max_position_embeddings,
      typeVocab: 0, norm: { eps: 1e-5, bias: false }, maskMultiply: false, project: false });
    assert.deepEqual(spec.attention, { kind: 'standard', bias: false, scale: spec.headDim ** -0.5,
      rope: { thetaGlobal: want.theta[0], thetaLocal: want.theta[1] }, window: { half: want.half, globalEvery: 3 } });
    assert.deepEqual(spec.block, { order: 'pre', firstNormIdentity: true, finalNorm: true,
      norm: { eps: 1e-5, bias: false } });
    assert.deepEqual(spec.ffn, { kind: 'geglu', act: 'gelu', bias: false });
    assert.deepEqual(head, want.head);
  });
}

test('ModernBERT: biases follow the config, the classifier always has one', () => {
  const e = entry('sheltron-ai/prompt-guard-68m');
  const { spec, head } = build(e, { attention_bias: true, mlp_bias: true, norm_bias: true, classifier_bias: true });
  assert.equal(spec.attention.bias, true);
  assert.equal(spec.ffn.bias, true);
  assert.deepEqual(spec.block.norm, { eps: 1e-5, bias: true });
  assert.deepEqual((head as { steps: object[] }).steps[0], denseK('head.dense', 512, 512, 'gelu', true));
  assert.deepEqual((head as { steps: object[] }).steps[1], { op: 'norm', name: 'head.norm', eps: 1e-5, bias: true });
});

test('ModernBERT: rope_parameters, the old theta keys and the defaults agree', () => {
  const e = entry('ibm-granite/granite-embedding-reranker-english-r2');
  const base = { ...e.config, global_rope_theta: undefined, local_rope_theta: undefined };
  const thetas = (patch: Record<string, unknown>) => {
    const { spec } = specFromHfConfig({ ...base, ...patch }, { task: e.task });
    return [spec.attention.rope!.thetaGlobal, spec.attention.rope!.thetaLocal];
  };
  assert.deepEqual(thetas({}), [160000, 10000]); // defaults of transformers 5.0.0
  assert.deepEqual(thetas({ global_rope_theta: 5, local_rope_theta: 7 }), [5, 7]);
  assert.deepEqual(thetas({ rope_parameters: { full_attention: { rope_theta: 9, rope_type: 'default' },
    sliding_attention: { rope_theta: 3 } }, global_rope_theta: 5 }), [9, 3]);
});

// One error per rejected key of the DeBERTa and ModernBERT rows; the message names the key.
const DEB = 'protectai/deberta-v3-base-prompt-injection-v2';
const MB = 'sheltron-ai/prompt-guard-68m';
const ERRORS_K286: [string, string, Record<string, unknown>, RegExp][] = [
  ['share_att_key false', DEB, { share_att_key: false }, /'share_att_key'/],
  ['conv_groups', DEB, { conv_kernel_size: 3, conv_groups: 2 }, /'conv_groups'/],
  ['even conv_kernel_size', DEB, { conv_kernel_size: 4 }, /'conv_kernel_size'/],
  ['embedding_size', DEB, { embedding_size: 128 }, /'embedding_size'/],
  ['pos_att_type with one type', DEB, { pos_att_type: ['c2p'] }, /'pos_att_type'/],
  ['pos_att_type with a foreign type', DEB, { pos_att_type: ['c2p', 'p2p'] }, /'pos_att_type'/],
  ['empty pos_att_type with relative attention', DEB, { pos_att_type: [] }, /'pos_att_type'/],
  ['norm_rel_ebd', DEB, { norm_rel_ebd: 'none' }, /'norm_rel_ebd'/],
  ['position_buckets', DEB, { position_buckets: -1 }, /'position_buckets'/],
  ['pooler_hidden_size', DEB, { pooler_hidden_size: 512 }, /'pooler_hidden_size'/],
  ['pooler_hidden_act', DEB, { pooler_hidden_act: 'gelu_new' }, /'pooler_hidden_act'/],
  ['hidden_act of DeBERTa', DEB, { hidden_act: 'gelu_new' }, /'hidden_act'/],
  ['DeBERTa head dimension', DEB, { num_attention_heads: 8 }, /head dimension 96 exceeds 64/],
  ['no position signal', 'xushijie/polyBERT', { position_biased_input: false }, /'position_biased_input'/],
  ['rope_scaling', MB, { rope_scaling: { rope_type: 'yarn', factor: 2 } }, /'rope_scaling'/],
  ['rope_type', MB, { rope_parameters: { full_attention: { rope_type: 'yarn', rope_theta: 1 },
    sliding_attention: { rope_theta: 1 } } }, /rope_type/],
  ['rope_parameters without layer types', MB, { rope_parameters: { rope_theta: 1 } }, /'rope_parameters'/],
  ['layer_types', MB, { global_attn_every_n_layers: 2 }, /'layer_types'/],
  ['hidden_activation', MB, { hidden_activation: 'silu' }, /'hidden_activation'/],
  ['classifier_pooling', MB, { classifier_pooling: 'max' }, /'classifier_pooling'/],
  ['classifier_activation', MB, { classifier_activation: 'gelu_new' }, /'classifier_activation'/],
  ['is_causal', MB, { is_causal: true }, /'is_causal'/],
  ['ModernBERT head dimension', MB, { num_attention_heads: 4 }, /head dimension 128 exceeds 64/],
];
for (const [name, id, patch, re] of ERRORS_K286) {
  test(`rejects ${name}`, () => assert.throws(() => build(entry(id), patch), re));
}
