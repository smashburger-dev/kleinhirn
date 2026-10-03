// K28.R group D: semantics of the list models (R12 multi-label scores, R09 pad-id rejection,
// R13 max pooling). R12 is the review test, unchanged. R09 and R13 are new contracts: the review
// tests ask for real HF positions (R09, deferred to K27, stays red in tests/review) and for the
// kernel source (R13), the ordered fixes are a rejection and a found flag. The recording mock does
// not run WGSL; the pool kernel is checked by its source and a transcription of its MODE 1 loop.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { specFromHfConfig } from '../src/plan/hf.ts';
import { padIdIndex } from '../src/plan/spec.ts';
import { model, source } from './helpers/review.mjs';

const models = JSON.parse(readFileSync('data/k28/models.json', 'utf8')).models;
const bert = (extra = {}) => ({
  model_type: 'bert', hidden_size: 16, num_hidden_layers: 1, num_attention_heads: 2,
  intermediate_size: 32, vocab_size: 8, ...extra,
});

// ---------- R12 ----------

test('R12 multi-label classify must not softmax logits', async () => {
  // Frozen config explicitly has problem_type=multi_label_classification, 28 labels.
  const { engine } = model('SamLowe/roberta-base-go_emotions');
  engine.labels = Array.from({ length: 28 }, (_, i) => String(i));
  engine.runIds = async () => ({ data: new Float32Array(28) });
  engine.textInput = () => ({ inputIds: [0] });
  // Isolate postprocessing. These logits are mocked, not measured from that text.
  const out = await engine.classify('two simultaneous emotions');
  assert.deepEqual([...out.scores], Array(28).fill(0.5));
});

async function classifyWith(problemType, logits, classes = logits.length) {
  const { spec, head } = specFromHfConfig(
    bert({ ...(problemType ? { problem_type: problemType } : {}), num_labels: classes }),
    { task: 'sequence-classification' });
  const { engine } = model();
  Object.assign(engine, { spec, head,
    labels: Array.from({ length: classes }, (_, i) => `L${i}`),
    runIds: async () => ({ data: Float32Array.from(logits) }),
    textInput: () => ({ inputIds: [0] }) });
  return engine.classify('x');
}

test('R12 classify: single-label softmax, multi-label sigmoid with the largest logit, regression raw', async () => {
  const single = await classifyWith(undefined, [0, 2, 1]);
  assert.equal(single.index, 1);
  assert.ok(Math.abs(single.scores.reduce((a, b) => a + b) - 1) < 1e-6);
  assert.equal(single.score, single.scores[1]);

  const multi = await classifyWith('multi_label_classification', [-3, 2, 1]);
  assert.equal(multi.index, 1);
  assert.equal(multi.label, 'L1');
  assert.ok(Math.abs(multi.score - 1 / (1 + Math.exp(-2))) < 1e-6);
  assert.ok(Math.abs(multi.scores[0] - 1 / (1 + Math.exp(3))) < 1e-6);
  assert.ok(multi.scores.reduce((a, b) => a + b) > 1, 'independent sigmoids do not sum to 1');

  const one = await classifyWith(undefined, [4.5]);
  assert.deepEqual([one.index, one.score, [...one.scores]], [0, 4.5, [4.5]]);
  const reg = await classifyWith('regression', [1.5, -2]);
  assert.equal(reg.index, 0);
  assert.equal(reg.score, 1.5);
  assert.deepEqual([...reg.scores], [1.5, -2]);
});

test('R12 the head carries problem_type, the default is left out', () => {
  const head = (extra, classes) => specFromHfConfig(bert({ ...extra, num_labels: classes }),
    { task: 'sequence-classification' }).head;
  assert.equal(head({ problem_type: 'multi_label_classification' }, 4).problem, 'multi');
  assert.equal(head({ problem_type: 'regression' }, 3).problem, 'regression');
  assert.equal(head({ problem_type: 'single_label_classification' }, 1).problem, 'single');
  assert.equal('problem' in head({ problem_type: 'single_label_classification' }, 3), false);
  assert.equal('problem' in head({ problem_type: 'regression' }, 1), false);
  assert.equal('problem' in head({}, 3), false);
  assert.throws(() => head({ problem_type: 'ordinal' }, 3), /problem_type/);
  const go = models.find((m) => m.id === 'SamLowe/roberta-base-go_emotions');
  const goHead = specFromHfConfig(go.config, { task: go.task,
    template: [go.config.bos_token_id, null, go.config.eos_token_id] }).head;
  assert.equal(goHead.problem, 'multi');
  assert.equal(goHead.classes, 28);
});

// ---------- R13 ----------

test('R13 pool max starts at the first valid row, no sentinel', () => {
  const src = source('src/kernels/pool.wgsl');
  assert.doesNotMatch(src, /60000/);
  assert.match(src, /var found = false/);
  // Transcription of the MODE 1 loop: select(v, max(best, v), found), then found = true.
  const pool = (mask, values) => {
    let best = 0, found = false;
    mask.forEach((m, i) => {
      if (m > 0.5) { best = found ? Math.max(best, values[i]) : values[i]; found = true; }
    });
    return best;
  };
  assert.equal(pool([1], [-64000]), -64000);
  assert.equal(pool([1, 1, 0], [-64000, -65000, 9]), -64000);
  assert.equal(pool([0, 0], [5, 6]), 0);
  assert.equal(pool([0, 1, 1], [9, -3, -1]), -1);
});

// ---------- R09 ----------

const ROBERTA = 'cardiffnlp/twitter-roberta-base-sentiment-latest';

test('R09 rule: no id may be the pad id; with offset equal to the pad id, id 0 must be it and no other', () => {
  const normal = { positionOffset: 2, padId: 1 };
  assert.equal(padIdIndex(normal, [0, 7, 8, 2]), -1);
  assert.equal(padIdIndex(normal, [0, 7, 1, 8, 2]), 2);
  assert.equal(padIdIndex(normal, [1]), 0);
  const special = { positionOffset: 0, padId: 0 }; // d0rj/e5-small-en-ru: <s> and <pad> are both 0
  assert.equal(padIdIndex(special, [0, 5, 6, 2]), -1);
  assert.equal(padIdIndex(special, [5, 6, 2]), 0);
  assert.equal(padIdIndex(special, [0, 5, 0, 2]), 2);
  assert.equal(padIdIndex({ positionOffset: 0 }, [1, 1, 1]), -1, 'a spec without a pad id is not checked');
});

test('R09 runIds and runIdsBatch throw naming the index of the pad id', async () => {
  const { engine, spec } = model(ROBERTA);
  assert.equal(spec.embed.padId, 1);
  await assert.rejects(() => engine.runIds({ inputIds: [0, 7, 1, 6, 2] }), /index 2 is the pad id 1/);
  await assert.rejects(() => engine.runIdsBatch([{ inputIds: [0, 2] }, { inputIds: [0, 7, 1, 2] }]),
    /input 1: .*index 2 is the pad id 1/);
  // ids without the pad id run, in a call with padding rows (3 callers pad up to 4)
  const outs = await engine.runIdsBatch([{ inputIds: [0, 3, 2] }, { inputIds: [0, 4, 2] }, { inputIds: [0, 5, 6, 2] }]);
  assert.equal(outs.length, 3);
});

test('R09 text methods reach the rule: a literal pad token in the text throws', async () => {
  const { engine } = model(ROBERTA);
  engine.textInput = () => ({ inputIds: [0, 7, 1, 8, 2] }); // the tokenizer maps "<pad>" to id 1
  await assert.rejects(() => engine.classify('a <pad> b'), /index 2 is the pad id 1/);
});

test('R09 special offset: runIds needs id 0 first and nowhere else', async () => {
  const { engine, spec } = model('d0rj/e5-small-en-ru', 'f32', [128]);
  assert.deepEqual([spec.embed.padId, spec.embed.positionOffset], [0, 0]);
  await assert.rejects(() => engine.runIds({ inputIds: [5, 6] }), /index 0 is not the pad id 0/);
  await assert.rejects(() => engine.runIds({ inputIds: [0, 5, 0, 2] }), /index 2 is the pad id 0/);
  assert.equal((await engine.runIds({ inputIds: [0, 5, 6, 2] })).seqLen, 4);
});

test('R09 BERT models are not checked', async () => {
  const { engine, spec } = model();
  assert.equal(spec.embed.padId, undefined);
  assert.equal((await engine.runIds({ inputIds: [1, 1, 1] })).seqLen, 3);
});

// HF positions of RoBERTa and XLM-R: cumsum(ids != pad) * (ids != pad) + pad.
function hfPositionsDiffer(ids, pad, offset) {
  let count = 0;
  return ids.some((id, i) => {
    const hf = id === pad ? pad : pad + ++count;
    return hf !== i + offset;
  });
}

// Gate: for every case with ids of every RoBERTa and XLM-R model with golden tokenizer cases, the
// rule throws exactly when the HF positions differ from the engine positions i + positionOffset
// (pad id and offset from specFromHfConfig on the frozen config and the single template <s> A </s>).
// Cases are not skipped by text. Skipped when models/ is absent.
test('R09 gate: the rule throws exactly where HF and engine positions differ (golden tokenizer cases)', (t) => {
  if (!existsSync('models/k28')) return t.skip('models/k28 is absent');
  let checkedModels = 0, checkedCases = 0, rejected = 0;
  for (const e of models.filter((m) => ['roberta', 'xlm-roberta'].includes(m.config.model_type))) {
    const dir = `models/k28/${e.id.replace('/', '__')}/golden`;
    if (!existsSync(dir)) continue;
    const files = readdirSync(dir).filter((f) => /^tokenizer-cases.*\.json$/.test(f));
    if (!files.length) continue;
    const { spec } = specFromHfConfig(e.config, { task: e.task, sentenceTransformers: e.sentenceTransformers,
      template: [e.config.bos_token_id, null, e.config.eos_token_id] });
    assert.notEqual(spec.embed.padId, undefined, e.id);
    checkedModels += 1;
    for (const f of files) {
      for (const [i, c] of JSON.parse(readFileSync(`${dir}/${f}`, 'utf8')).cases.entries()) {
        if (!c.ok || !Array.isArray(c.input_ids)) continue;
        checkedCases += 1;
        const throws = padIdIndex(spec.embed, c.input_ids) >= 0;
        if (throws) rejected += 1;
        assert.equal(throws, hfPositionsDiffer(c.input_ids, spec.embed.padId, spec.embed.positionOffset),
          `${e.id} ${f} case ${i}`);
      }
    }
  }
  assert.ok(checkedModels > 0 && checkedCases > 0);
  console.log(`R09 gate: ${checkedModels} models, ${checkedCases} cases, ${rejected} rejected`);
});
