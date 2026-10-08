// K28.R group D: semantics of the list models (R12 multi-label scores, R09 pad-id rejection,
// R13 max pooling). R12 is the review test, unchanged. R09 and R13 are new contracts: the review
// tests ask for real HF positions (R09, deferred to K27, stays red in tests/review) and for the
// kernel source (R13), the ordered fixes are a rejection and a found flag. The recording mock does
// not run WGSL; the pool kernel is checked by its source and a transcription of its MODE 1 loop.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { specFromHfConfig } from '../src/plan/hf.ts';
import { positionRows } from '../src/plan/spec.ts';
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

test('R09 positions follow transformers: a pad id keeps the pad row, other ids count', () => {
  assert.deepEqual(positionRows(1, [0, 7, 8, 2]), [2, 3, 4, 5]);
  assert.deepEqual(positionRows(1, [0, 7, 1, 8, 2]), [2, 3, 1, 4, 5]);
  assert.deepEqual(positionRows(1, [1]), [1]);
  // d0rj/e5-small-en-ru: <s> and <pad> are both 0
  assert.deepEqual(positionRows(0, [0, 5, 6, 2]), [0, 1, 2, 3]);
  assert.deepEqual(positionRows(0, [5, 6, 2]), [1, 2, 3]);
  assert.deepEqual(positionRows(0, [0, 5, 0, 2]), [0, 1, 0, 2]);
});

test('R09 runIds and runIdsBatch run ids with the pad id, positions packed in the high 16 bits', async () => {
  const { engine, spec } = model(ROBERTA);
  assert.equal(spec.embed.padId, 1);
  assert.equal((await engine.runIds({ inputIds: [0, 7, 1, 6, 2] })).seqLen, 5);
  const outs = await engine.runIdsBatch([{ inputIds: [0, 2] }, { inputIds: [0, 7, 1, 2] }, { inputIds: [0, 5, 6, 2] }]);
  assert.equal(outs.length, 3);
  const { typeIds } = engine.maskAndTypes([{ inputIds: [0, 7, 1, 6, 2] }], 8);
  assert.deepEqual(Array.from(typeIds, (t) => t >>> 16), [2, 3, 1, 4, 5, 1, 1, 1]);
  assert.deepEqual(Array.from(typeIds, (t) => t & 0xffff), [0, 0, 0, 0, 0, 0, 0, 0]);
});

test('R09 text methods run a literal pad token in the text', async () => {
  const { engine } = model(ROBERTA);
  engine.textInput = () => ({ inputIds: [0, 7, 1, 6, 2] }); // the tokenizer maps "<pad>" to id 1
  await assert.doesNotReject(() => engine.classify('a <pad> b'));
});

test('R09 special offset: ids without the pad id first run with transformers positions', async () => {
  const { engine, spec } = model('d0rj/e5-small-en-ru', 'f32', [128]);
  assert.deepEqual([spec.embed.padId, spec.embed.positionOffset], [0, 0]);
  assert.equal((await engine.runIds({ inputIds: [5, 6] })).seqLen, 2);
  assert.equal((await engine.runIds({ inputIds: [0, 5, 0, 2] })).seqLen, 4);
  assert.equal((await engine.runIds({ inputIds: [0, 5, 6, 2] })).seqLen, 4);
});

test('R09 BERT models are not checked', async () => {
  const { engine, spec } = model();
  assert.equal(spec.embed.padId, undefined);
  assert.equal((await engine.runIds({ inputIds: [1, 1, 1] })).seqLen, 3);
});

// Gate: for every case with ids of every RoBERTa and XLM-R model with golden tokenizer cases, the
// packed position rows equal transformers (cumsum(ids != pad) * (ids != pad) + pad), and they equal
// the rows before R09 (i + positionOffset) in every case without a pad id, so those bits stay.
// Skipped when models/ is absent.
test('R09 gate: positions equal transformers on the golden tokenizer cases, and the old rows without a pad id', (t) => {
  if (!existsSync('models/k28')) return t.skip('models/k28 is absent');
  let checkedModels = 0, checkedCases = 0, withPad = 0;
  for (const e of models.filter((m) => ['roberta', 'xlm-roberta'].includes(m.config.model_type))) {
    const dir = `models/k28/${e.id.replace('/', '__')}/golden`;
    if (!existsSync(dir)) continue;
    const files = readdirSync(dir).filter((f) => /^tokenizer-cases.*\.json$/.test(f));
    if (!files.length) continue;
    const { spec } = specFromHfConfig(e.config, { task: e.task, sentenceTransformers: e.sentenceTransformers,
      template: [e.config.bos_token_id, null, e.config.eos_token_id] });
    const { padId, positionOffset } = spec.embed;
    assert.notEqual(padId, undefined, e.id);
    checkedModels += 1;
    for (const f of files) {
      for (const [i, c] of JSON.parse(readFileSync(`${dir}/${f}`, 'utf8')).cases.entries()) {
        if (!c.ok || !Array.isArray(c.input_ids)) continue;
        checkedCases += 1;
        const rows = positionRows(padId, c.input_ids);
        let count = 0;
        assert.deepEqual(rows, c.input_ids.map((id) => (id === padId ? padId : padId + ++count)), `${e.id} ${f} case ${i}`);
        const old = c.input_ids.map((_, k) => k + positionOffset);
        if (rows.some((r, k) => r !== old[k])) withPad += 1;
        else assert.deepEqual(rows, old);
      }
    }
  }
  assert.ok(checkedModels > 0 && checkedCases > 0);
  console.log(`R09 gate: ${checkedModels} models, ${checkedCases} cases, ${withPad} with positions that change`);
});
