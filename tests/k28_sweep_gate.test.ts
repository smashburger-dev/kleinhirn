import assert from 'node:assert/strict';
import { test } from 'node:test';
import { evaluateGate, missedOnlyLogitTolerance, spanScoreTolerance } from '../tools/k28_sweep_gate.mjs';

const forwardOk = { task: 'token-classification', g2_pass: true, g3_pass: true, max_abs_logit_diff: 1e-5, argmax_per_token_agreement: 1 };
const p = (metrics: object, g4 = { rows: 200, differingRows: 0, groups: 50 }) => ({ metrics, g4 });
const text = { metrics: { textStats: { inputMismatch: 0, bitDiffRows: 0 }, spanExactAgreement: 1 } };
const base = {
  task: 'token-classification', forward: forwardOk,
  p32: p({ argmaxPerTokenAgreement: 1 }), p16: p({ argmaxPerTokenAgreement: 0.9998, maxAbsLogitDiff: 0.09 }), text,
  sim: { delta_sim: 0.1 },
  close: { decisions: { tokenArgmax: { n: 6350, close: 11, failures: 1, failuresNotClose: 0 } } },
};

test('a model with matching decisions and one close f16 miss passes', () => {
  assert.deepEqual(evaluateGate(base), { pass: true, reasons: [] });
});

test('a not close f16 miss on 20 decisions fails the f16 gate', () => {
  const g = evaluateGate({ ...base, task: 'reranking', p16: p({ bestPassageAgreement: 0.95, maxAbsLogitDiff: 0.09 }),
    close: { decisions: { bestPassage: { n: 20, close: 1, failures: 1, failuresNotClose: 1 } } } });
  assert.equal(g.pass, false);
  assert.match(g.reasons[0], /bestPassage/);
});

test('f32 below 100 percent, a batch row difference and a text difference each fail', () => {
  assert.equal(evaluateGate({ ...base, p32: p({ argmaxPerTokenAgreement: 0.9999 }) }).pass, false);
  assert.equal(evaluateGate({ ...base, p16: p({}, { rows: 200, differingRows: 1, groups: 50 }) }).pass, false);
  assert.equal(evaluateGate({ ...base, text: { metrics: { textStats: { inputMismatch: 0, bitDiffRows: 2 }, spanExactAgreement: 1 } } }).pass, false);
});

test('float64 fallback only when the numpy check missed the logit tolerance alone', () => {
  const miss = { ...forwardOk, g2_pass: false, max_abs_logit_diff: 3e-4 };
  assert.equal(missedOnlyLogitTolerance(miss), true);
  assert.equal(evaluateGate({ ...base, forward: miss, f64: { numpy_vs_torch64: 2e-5 } }).pass, true);
  assert.equal(evaluateGate({ ...base, forward: miss, f64: { numpy_vs_torch64: 2e-4 } }).pass, false);
  assert.equal(evaluateGate({ ...base, forward: miss }).pass, false);
  assert.equal(missedOnlyLogitTolerance({ ...miss, argmax_per_token_agreement: 0.99 }), false);
  assert.equal(missedOnlyLogitTolerance({ ...miss, task: 'embeddings' }), false);
});

test('f16 is evaluable only up to three times the simulated deviation', () => {
  const at3 = evaluateGate({ ...base, p16: p({ argmaxPerTokenAgreement: 0.9998, maxAbsLogitDiff: 0.3 }) });
  assert.equal(at3.pass, true);
  const over = evaluateGate({ ...base, p16: p({ argmaxPerTokenAgreement: 1, maxAbsLogitDiff: 0.31 }) });
  assert.equal(over.pass, false);
  assert.match(over.reasons[0], /f16 weicht stärker ab als die Simulation/);
  // no simulation (f16 manifest deleted): kept only for an old run without wrong decisions and delta_run <= 0.05
  const noWrong = { decisions: { tokenArgmax: { n: 6350, close: 11, failures: 0, failuresNotClose: 0 } } };
  assert.equal(evaluateGate({ ...base, sim: undefined, close: noWrong,
    p16: p({ argmaxPerTokenAgreement: 1, maxAbsLogitDiff: 0.04 }) }).pass, true);
  const none = evaluateGate({ ...base, sim: undefined });
  assert.equal(none.pass, false);
  assert.match(none.reasons.join(), /f16 simulation missing/);
  assert.equal(evaluateGate({ ...base, sim: undefined, close: noWrong,
    p16: p({ argmaxPerTokenAgreement: 1, maxAbsLogitDiff: 0.06 }) }).pass, false);
});

const emb = {
  task: 'embeddings', forward: { task: 'embeddings', g2_pass: true, g3_pass: true },
  p32: p({ minCosineFinal: 0.99999 }), text,
};

test('embeddings: cosine 0.999 passes, a miss the simulation reproduces is f16 storage and recommends f32', () => {
  const ok = evaluateGate({ ...emb, p16: p({ minCosineFinal: 0.9995 }), sim: { delta_sim: 0.1, min_cosine_final: 0.9996 } });
  assert.deepEqual(ok, { pass: true, reasons: [] });
  const stored = evaluateGate({ ...emb, p16: p({ minCosineFinal: 0.99816 }), sim: { delta_sim: 0.3, min_cosine_final: 0.9982 } });
  assert.equal(stored.pass, false);
  assert.equal(stored.recommendedPrecision, 'f32');
  assert.match(stored.reasons.join(), /f16-Speicherung, simuliert/);
});

test('embeddings: a miss the simulation does not reproduce is a stronger deviation, no recommendation', () => {
  const g = evaluateGate({ ...emb, p16: p({ minCosineFinal: 0.99816 }), sim: { delta_sim: 0.01, min_cosine_final: 0.99999 } });
  assert.equal(g.pass, false);
  assert.equal(g.recommendedPrecision, undefined);
  assert.match(g.reasons.join(), /f16 weicht stärker ab als die Simulation/);
});

test('f32 deviations are allowed only inside the golden error, and only with a float64 reference', () => {
  const miss = { ...forwardOk, g2_pass: false, max_abs_logit_diff: 3e-4 };
  const dev = (notAllowed: number) => ({ ...base.close,
    decisions32: { limit: 1.6e-3, decisions: { tokenArgmax: { n: 6350, deviations: 3, allowed: 3 - notAllowed, notAllowed } } } });
  const args = { ...base, forward: miss, f64: { numpy_vs_torch64: 2e-5 }, p32: p({ argmaxPerTokenAgreement: 0.9995 }) };
  assert.equal(evaluateGate({ ...args, close: dev(0) }).pass, true);
  const bad = evaluateGate({ ...args, close: dev(1) });
  assert.equal(bad.pass, false);
  assert.match(bad.reasons.join(), /f32 tokenArgmax 1 of 6350/);
  // without a float64 reference the rule does not apply: 100 % stays
  assert.equal(evaluateGate({ ...base, p32: p({ argmaxPerTokenAgreement: 0.9995 }), close: dev(0) }).pass, false);
});

test('span scores may differ by the inexact golden\'s own error, the group and offsets stay exact', () => {
  const miss = { ...forwardOk, g2_pass: false, max_abs_logit_diff: 3e-4 };
  const f64 = { numpy_vs_torch64: 2e-5, golden32_vs_torch64: 8.3e-4 };
  const span = (score: number, group = 'A') => ({ group, score, start: 0, end: 4 });
  const want = [{ entity_group: 'A', score: 0.5574, start: 0, end: 4 }];
  const miss1 = (got: object[]) => ({ metrics: { textStats: { inputMismatch: 0, bitDiffRows: 0 },
    spanExactAgreement: 0.995, spanTexts: 200, firstDifferingText: { got, want } } });
  assert.equal(spanScoreTolerance(miss, f64), 8.3e-4);
  assert.equal(spanScoreTolerance(forwardOk, f64), 1e-4);
  assert.equal(spanScoreTolerance(miss, undefined), 1e-4);
  const run = (text: object, f = f64) => evaluateGate({ ...base, forward: miss, f64: f, text }).pass;
  assert.equal(run(miss1([span(0.5573)])), true);
  assert.equal(run(miss1([span(0.5560)])), false);
  assert.equal(run(miss1([span(0.5573, 'B')])), false);
  assert.equal(run(miss1([{ ...span(0.5573), end: 5 }])), false);
  assert.equal(run(miss1([span(0.5573)]), { numpy_vs_torch64: 2e-5, golden32_vs_torch64: 5e-5 }), false);
  // two missed texts cannot be re-checked from the stored first one
  assert.equal(run({ metrics: { ...miss1([span(0.5573)]).metrics, spanExactAgreement: 0.99 } }), false);
});
