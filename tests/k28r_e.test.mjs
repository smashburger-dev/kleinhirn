// K28.R group E: hangs and inputs (T03, R23, R14, R15, R27). The review tests move here
// unchanged (T03 as written: the loader now throws instead of hanging, which the original also
// accepts; R14 x2, R15, R23, R27), plus tests for the ordered contract: T03 rejects the empty
// pattern at load and survives a zero-length match, labels are completed at load, the prepared
// ID paths of GLiNER and Julia check ids.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { buildPlan } from '../src/plan/build.ts';
import { Kleinhirn } from '../src/index.ts';
import { JuliaEngine } from '../src/julia.ts';
import { JsonTokenizer } from '../src/tokenizer/hf/index.ts';
import { argmaxRows, completeLabels, tokenRowOffset } from '../src/tasks.ts';
import { model } from './helpers/review.mjs';
import { loadTiny } from './helpers/tiny-model.mjs';

const fixtureJson = (name) => JSON.parse(readFileSync(`tests/fixtures/hf-tokenizers/${name}.json`, 'utf8'));

// ---------- T03 ----------

test('T03 normalized added token which becomes empty must not hang', () => {
  const code = `
    import { readFileSync } from 'node:fs';
    import { JsonTokenizer } from './src/tokenizer/hf/index.ts';
    const j = JSON.parse(readFileSync('tests/fixtures/hf-tokenizers/bpe-nfc-bytelevel-post.json'));
    j.normalizer = {type:'BertNormalizer',clean_text:true,handle_chinese_chars:false,strip_accents:false,lowercase:false};
    j.added_tokens = [{id:999, content:'\\u0000', normalized:true, special:false}];
    const tokenizer = JsonTokenizer.fromJson(j);
    process.stderr.write('imports-complete; entering encode\\n');
    tokenizer.encode('x');
  `;
  // Run separately: a synchronous infinite regex loop cannot be interrupted by node:test timeout.
  let error;
  try {
    execFileSync(process.execPath, ['--import', './tests/helpers/node-hooks.mjs', '--input-type=module', '-e', code],
      { timeout: 1500, stdio: 'pipe' });
  } catch (e) { error = e; }
  if (error?.code === 'ETIMEDOUT') assert.match(String(error.stderr), /imports-complete; entering encode/);
  assert.notEqual(error?.code, 'ETIMEDOUT', 'encode did not terminate in 1500ms');
});

test('T03 an added token that normalizes to nothing is rejected at load, with its id', () => {
  const j = fixtureJson('bpe-nfc-bytelevel-post');
  j.normalizer = { type: 'BertNormalizer', clean_text: true, handle_chinese_chars: false,
    strip_accents: false, lowercase: false };
  j.added_tokens = [{ id: 999, content: '\u0000', normalized: true, special: false }];
  assert.throws(() => JsonTokenizer.fromJson(j), /added token 999 is empty after normalization/);
  // the same token matched on the raw text (normalized false) is a normal token
  j.added_tokens[0].normalized = false;
  assert.doesNotThrow(() => JsonTokenizer.fromJson(j));
});

// ---------- R23 ----------

test('R23 absent labels must not hang token classification', () => {
  const code = `
    import { model, fixture } from './tests/helpers/review.mjs';
    import { JsonTokenizer } from './src/tokenizer/hf/index.ts';
    const {engine} = model();
    engine.head = {type:'token', classes:2, steps:[{
      op:'dense', name:'head.classifier', in:engine.spec.hidden, out:2, act:'none', bias:true,
    }]};
    engine.tokenizer = JsonTokenizer.fromJson(fixture('bpe-roberta'));
    engine.runIds = async (input) => ({data:new Float32Array(input.inputIds.length*2),rows:input.inputIds.length,cols:2});
    process.stderr.write('imports-complete; entering tokenClassify\\n');
    await engine.tokenClassify('hello');
  `;
  let error;
  try { execFileSync(process.execPath, ['--import', './tests/helpers/node-hooks.mjs', '--input-type=module', '-e', code],
    { timeout: 1500, stdio: 'pipe' }); } catch (e) { error = e; }
  if (error?.code === 'ETIMEDOUT') assert.match(String(error.stderr), /imports-complete; entering tokenClassify/);
  assert.notEqual(error?.code, 'ETIMEDOUT', 'argmaxRows(data, labels.length=0) did not terminate');
});

test('R23 argmaxRows throws on a column count that is not a positive integer', () => {
  for (const cols of [0, -1, 1.5, NaN, Infinity]) {
    assert.throws(() => argmaxRows(new Float32Array(4), cols), /positive integer/, String(cols));
  }
  assert.deepEqual(argmaxRows(Float32Array.of(1, 3, 2, 9, 0, 0), 3), [1, 0]);
});

test('R23 labels: LABEL_i for missing ids, ids from 0 to classes - 1, no more labels than classes', () => {
  assert.deepEqual(completeLabels({ 0: 'O', 2: 'B-X' }, 3), ['O', 'LABEL_1', 'B-X']);
  assert.deepEqual(completeLabels(undefined, 2), ['LABEL_0', 'LABEL_1']);
  assert.throws(() => completeLabels({ 0: 'a', 1: 'b', 2: 'c' }, 2), /label id '2'/);
  assert.throws(() => completeLabels({ 1: 'a', '-1': 'b' }, 2), /label id '-1'/);
  assert.throws(() => completeLabels({ x: 'a' }, 2), /label id 'x'/);
  assert.throws(() => completeLabels({ '01': 'a' }, 2), /label id '01'/);
});

test('R23 EncoderModel.load keeps the manifest labels of a token head', async (t) => {
  const loaded = await loadTiny({ setup: undefined });
  t.after(loaded.restore);
  assert.deepEqual(loaded.engine.labels, ['O', 'X']);
});

// ---------- R14 ----------

test('R14 public runIds must reject fractional token ids', async () => {
  const { engine } = model();
  await assert.rejects(() => engine.runIds({ inputIds: [0.5] }), /integer|token id/);
});

test('R14 public runIds must reject NaN type ids', async () => {
  const { engine } = model('cardiffnlp/twitter-roberta-base-sentiment-latest');
  await assert.rejects(() => engine.runIds({ inputIds: [0], typeIds: [NaN] }), /type/);
});

test('R14 token ids: NaN, negative, fractional and beyond the vocabulary throw; type ids must be integers', async () => {
  const { engine } = model();
  for (const id of [NaN, -1, 0.5, 8, Infinity]) {
    await assert.rejects(() => engine.runIds({ inputIds: [id] }), /token id/, String(id));
  }
  await assert.rejects(() => engine.runIds({ inputIds: [0, 1], typeIds: [0, 0.5] }), /type/);
  await assert.rejects(() => engine.runIds({ inputIds: [0, 1], typeIds: [0, 2] }), /type/);
  assert.equal((await engine.runIds({ inputIds: [0, 7], typeIds: [0, 1] })).seqLen, 2);
});

test('R14 the prepared id paths of GLiNER and Julia check ids, valid ids give the same rows', () => {
  const hidden = 4;
  const table = Float32Array.from({ length: 3 * hidden }, (_, i) => i + 1); // vocabulary of 3 rows
  const weights = { embeddings: table };
  const gliner = Reflect.construct(Kleinhirn, [{}, weights, null, { hiddenSize: hidden }, 1, 'f32']);
  const julia = Reflect.construct(JuliaEngine, [{}, weights, null, { hiddenSize: hidden }, 'f32']);
  const plan = { length: 4, batch: 2 };
  for (const engine of [gliner, julia]) {
    const ok = { inputIds: Int32Array.of(2, 0), seqLen: 2 };
    const rows = engine.embeddingRows(ok, plan);
    assert.deepEqual([...rows.slice(0, 8)], [9, 10, 11, 12, 1, 2, 3, 4]);
    const batch = engine.batchEmbeddingRows([ok, ok], plan);
    assert.deepEqual([...batch.slice(0, 8)], [9, 10, 11, 12, 1, 2, 3, 4]);
    for (const id of [3, -1, 0.5, NaN]) {
      const bad = { inputIds: Float32Array.of(id), seqLen: 1 };
      assert.throws(() => engine.embeddingRows(bad, plan), /token id/, `embeddingRows ${id}`);
      assert.throws(() => engine.batchEmbeddingRows([bad], plan), /token id/, `batchEmbeddingRows ${id}`);
    }
    // f16 tables take the same check
    engine.weights = { embeddings: new Uint16Array(table.length) };
    assert.throws(() => engine.embeddingRows({ inputIds: Int32Array.of(3), seqLen: 1 }, plan), /token id/);
  }
  assert.equal(tokenRowOffset(2, hidden, table.length), 8);
});

// ---------- R15 ----------

test('R15 mixed empty batch input must behave like empty single input', async () => {
  const { engine } = model();
  await assert.rejects(() => engine.runIdsBatch([{ inputIds: [] }, { inputIds: [0] }]), /empty input/);
});

test('R15 an empty caller row throws before any upload, a short batch (internal padding rows) still runs', async () => {
  const { engine, gpu } = model();
  const before = gpu.log.length;
  await assert.rejects(() => engine.runIdsBatch([{ inputIds: [0] }, { inputIds: [] }, { inputIds: [1] }]),
    /empty input \(input 1\)/);
  assert.equal(gpu.log.length, before, 'nothing was uploaded or submitted');
  const outs = await engine.runIdsBatch([{ inputIds: [0] }, { inputIds: [1, 2] }, { inputIds: [3] }]);
  assert.equal(outs.length, 3);
});

// ---------- R27 ----------

test('R27 embed head ending in norm must retain the Dense output width', () => {
  const { engine, spec } = model();
  engine.head = { type: 'embed', pool: 'mean', normalize: false, steps: [
    { op: 'dense', name: 'head.dense0', in: spec.hidden, out: 2, act: 'none', bias: true },
    { op: 'norm', name: 'head.norm', eps: 1e-5, bias: false },
  ] };
  // buildPlan accepts this public HeadSpec; output is B4 x 2.
  const p = buildPlan(spec, engine.head, { length: 128, batch: 4, markers: 0, f16: false });
  assert.equal(p.output.cols, 2);
  const result = engine.shape(Float32Array.of(1, 1, 2, 2, 3, 3, 4, 4), 1, 1, { length: 128 });
  assert.equal(result.cols, 2);
  assert.deepEqual([...result.data], [2, 2]);
});

test('R27 the engine width equals the plan width for every embed head shape', () => {
  const { engine, spec } = model();
  const dense = (i, o) => ({ op: 'dense', name: `head.dense${i}`, in: i, out: o, act: 'none', bias: true });
  const norm = { op: 'norm', name: 'head.norm', eps: 1e-5, bias: false };
  for (const steps of [[], [norm], [dense(spec.hidden, 5)], [dense(spec.hidden, 5), norm],
    [dense(spec.hidden, 5), norm, dense(5, 3)], [norm, dense(spec.hidden, 7), norm]]) {
    engine.head = { type: 'embed', pool: 'mean', normalize: false, steps };
    const p = buildPlan(spec, engine.head, { length: 128, batch: 1, markers: 0, f16: false });
    assert.equal(engine.cols, p.output.cols, JSON.stringify(steps.map((s) => s.op)));
  }
});
