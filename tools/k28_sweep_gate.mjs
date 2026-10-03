// K28.S gate rules and result collection for the sweep driver (tools/k28_sweep.mjs).
// Pure functions over the result files of convert/k28_forward.py, bench/run-k28-parity.mjs and
// convert/k28_close_calls.py. The rules are the ones of docs/K28_4_WORKORDER.md (G2, G3),
// docs/K28_5_WORKORDER.md (G2) and K28_DESIGN section 10 (close decisions, f64 fallback), with the f16 and
// f32 rules of docs/K28_S3_WORKORDER.md (decisions 3 and 4).

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

export const LOGIT_TOL = 1e-4;
export const COSINE_F32 = 0.9999;
export const COSINE_F16 = 0.999;
export const F16_RATE = 0.995;
// f16 run vs independent simulation: delta_run <= SIM_FACTOR * delta_sim, or the run is not evaluable.
export const SIM_FACTOR = 3;
export const REASON_F16_STRONGER = 'f16 weicht stärker ab als die Simulation';
export const REASON_F16_STORAGE = 'f16-Speicherung, simuliert';

const RATE_KEYS = ['argmax_agreement', 'zero_shot_agreement', 'best_passage_agreement',
  'argmax_per_token_agreement'];

// True when the numpy check missed only the absolute logit tolerance: every decision matches and the
// f16 tensors are bit equal. Only then the float64 fallback applies (embeddings have no torch64 script).
export function missedOnlyLogitTolerance(forward) {
  if (!forward || forward.g2_pass) return false;
  if (forward.task === 'embeddings') return false;
  if (!(forward.max_abs_logit_diff > LOGIT_TOL)) return false;
  return forward.g3_pass === true
    && RATE_KEYS.every((k) => forward[k] === undefined || forward[k] === 1);
}

// Slim a parity result for the sweep result file: drop the per-text arrays of the text run.
export function slimParity(result) {
  if (!result) return result;
  const m = { ...(result.metrics ?? {}) };
  // firstDifferingText stays: the span gate re-checks a single differing text against its own tolerance.
  for (const k of ['choices', 'spansGot']) delete m[k];
  const { task, model, precision, commit, run_id: runId, browser, loadAvgStart, loadAvgEnd } = result;
  return { model, precision, task, commit, runId, browser, loadAvgStart, loadAvgEnd, metrics: m, g4: result.g4,
    gpuErrors: result.gpuErrors };
}

const RATE_METRICS = [
  ['argmax', 'argmaxAgreement'], ['zeroShot', 'zeroShotAgreement'],
  ['bestPassage', 'bestPassageAgreement'], ['tokenArgmax', 'argmaxPerTokenAgreement'],
];

// f32: 100 % against the fp32 golden. Where the fp32 golden is itself inexact (a float64 reference exists),
// a deviation counts only when its torch64 gap is not below twice the golden's error (close.decisions32).
function checkF32(task, p32, close, f64, reasons) {
  if (!p32) { reasons.push('f32 parity missing'); return; }
  const m = p32.metrics ?? {};
  const d32 = f64 ? close?.decisions32 : undefined;
  for (const [name, key] of RATE_METRICS) {
    const v = m[key];
    if (typeof v !== 'number' || v === 1) continue;
    const d = d32?.decisions?.[name];
    if (d && typeof d.notAllowed === 'number') {
      if (d.notAllowed > 0) reasons.push(`f32 ${name} ${d.notAllowed} of ${d.n} deviations outside the golden's own error`);
    } else reasons.push(`f32 ${key} ${v}`);
  }
  if (task === 'embeddings' && !(m.minCosineFinal >= COSINE_F32)) reasons.push(`f32 minCosineFinal ${m.minCosineFinal}`);
  if (p32.g4 && p32.g4.differingRows !== 0) reasons.push(`f32 batch rows differ ${p32.g4.differingRows}`);
}

// A model whose f16 manifest was deleted after an old pass has no simulation. Its old run is kept when it has no
// wrong decision and delta_run <= LEGACY_DELTA (K28.S3 step 5: such models are reloaded only above that).
export const LEGACY_DELTA = 0.05;
function legacyF16Ok(task, m, close) {
  if (task === 'embeddings') return m.minCosineFinal >= COSINE_F16 && !(m.maxAbsDiffFinal > LEGACY_DELTA);
  const wrong = Object.values(close?.decisions ?? {}).reduce((sum, d) => sum + (d.failures ?? 0), 0);
  return !!close && wrong === 0 && m.maxAbsLogitDiff <= LEGACY_DELTA;
}

// f16: only evaluable when delta_run <= 3 * delta_sim (independent simulation). Embeddings: cosine >= 0.999;
// a miss that the simulation reproduces (1 - cos_run <= 3 * (1 - cos_sim)) is f16 storage, simulated, and the
// manifest then recommends f32. Other tasks: 99.5 % of the decisions that are not close (2 * delta_run).
// Returns { recommendedPrecision } or undefined.
function checkF16(task, p16, close, sim, reasons) {
  if (!p16) { reasons.push('f16 parity missing'); return undefined; }
  const m = p16.metrics ?? {};
  let recommendedPrecision;
  if (p16.g4 && p16.g4.differingRows !== 0) reasons.push(`f16 batch rows differ ${p16.g4.differingRows}`);
  if (!sim || !(sim.delta_sim >= 0)) {
    if (!legacyF16Ok(task, m, close)) reasons.push('f16 simulation missing');
    return undefined;
  }
  if (task === 'embeddings') {
    if (m.minCosineFinal >= COSINE_F16) return undefined;
    const lossRun = 1 - m.minCosineFinal;
    if (typeof sim.min_cosine_final === 'number' && lossRun <= SIM_FACTOR * (1 - sim.min_cosine_final)) {
      reasons.push(`f16 minCosineFinal ${m.minCosineFinal}: ${REASON_F16_STORAGE} (min cosine ${sim.min_cosine_final})`);
      recommendedPrecision = 'f32';
    } else reasons.push(`f16 minCosineFinal ${m.minCosineFinal}: ${REASON_F16_STRONGER}`);
    return recommendedPrecision;
  }
  const dRun = m.maxAbsLogitDiff;
  if (!(dRun <= SIM_FACTOR * sim.delta_sim)) {
    reasons.push(`${REASON_F16_STRONGER} (delta_run ${dRun}, delta_sim ${sim.delta_sim})`);
    return undefined;
  }
  if (!close) { reasons.push('close calls missing'); return undefined; }
  for (const [name, d] of Object.entries(close.decisions ?? {})) {
    const notClose = d.n - d.close;
    const wrong = d.failuresNotClose ?? d.failuresNotCloseAtLeast ?? 0;
    if (notClose > 0 && (notClose - wrong) / notClose < F16_RATE) reasons.push(`f16 ${name} ${wrong} of ${notClose} not close wrong`);
  }
  return undefined;
}

// Score tolerance of the text spans (decision of K28.6c). The browser run compares scores with 1e-4 against the
// fp32 golden. Where the fp32 golden is itself inexact (the float64 fallback applies), a score may differ by the
// golden's own error e = |golden32 - torch64| instead: max(1e-4, e). Group, start and end stay exact.
export const SPAN_SCORE_TOL = 1e-4;
export function spanScoreTolerance(forward, f64) {
  if (!f64 || !missedOnlyLogitTolerance(forward)) return SPAN_SCORE_TOL;
  const e = f64.golden32_vs_torch64;
  return typeof e === 'number' ? Math.max(SPAN_SCORE_TOL, e) : SPAN_SCORE_TOL;
}

// True when a differing text of the browser run differs only in span scores within tol: same spans, same
// group, same offsets. The stored result keeps only the first differing text, so only a miss of exactly one
// text can be re-checked here; any other miss stays a miss.
function onlyScoreWithin(text, tol) {
  const first = text.metrics?.firstDifferingText;
  const misses = Math.round((1 - text.metrics.spanExactAgreement) * (text.metrics.spanTexts ?? 0));
  if (misses !== 1 || !first?.got || !first?.want || first.got.length !== first.want.length) return false;
  return first.got.every((g, k) => {
    const w = first.want[k];
    return g.group === w.entity_group && g.start === w.start && g.end === w.end
      && Math.abs(g.score - w.score) <= tol;
  });
}

function checkText(task, text, reasons, tol) {
  if (!text) { reasons.push('text path missing'); return; }
  const s = text.metrics?.textStats;
  if (!s) { reasons.push('text path without stats'); return; }
  if (s.inputMismatch !== 0) reasons.push(`text inputMismatch ${s.inputMismatch}`);
  if (s.bitDiffRows !== 0) reasons.push(`text bitDiffRows ${s.bitDiffRows}`);
  if (task === 'token-classification' && text.metrics.spanExactAgreement !== 1
    && !(tol > SPAN_SCORE_TOL && onlyScoreWithin(text, tol))) {
    reasons.push(`spanExactAgreement ${text.metrics.spanExactAgreement}`);
  }
}

// Gate of one model: pass, the reasons of a miss, and recommendedPrecision 'f32' when f16 storage alone
// explains an f16 embedding miss.
export function evaluateGate({ task, forward, f64, p32, p16, text, close, sim }) {
  const reasons = [];
  if (!forward) reasons.push('numpy check missing');
  else if (!forward.g2_pass || !forward.g3_pass) {
    if (missedOnlyLogitTolerance(forward) && f64) {
      if (!(f64.numpy_vs_torch64 <= LOGIT_TOL)) reasons.push(`numpy vs torch64 ${f64.numpy_vs_torch64}`);
    } else {
      reasons.push(`numpy check failed (g2 ${forward.g2_pass}, g3 ${forward.g3_pass}, logit ${forward.max_abs_logit_diff ?? ''})`);
    }
  }
  checkF32(task, p32, close, f64, reasons);
  const recommendedPrecision = checkF16(task, p16, close, sim, reasons);
  checkText(task, text, reasons, spanScoreTolerance(forward, f64));
  return { pass: reasons.length === 0, reasons, ...(recommendedPrecision ? { recommendedPrecision } : {}) };
}

// Newest parity result file of a model: the run_id ends with the epoch milliseconds of the run.
export function latestParity(resultsDir, slug, precision, kind) {
  const re = new RegExp(`^k28-parity-${slug.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-${precision}-[0-9a-f]{7}(-text|-layers)?\\.json$`);
  let best = null;
  for (const name of readdirSync(resultsDir)) {
    const m = re.exec(name);
    if (!m) continue;
    const isText = m[1] === '-text';
    if (m[1] === '-layers' || isText !== (kind === 'text')) continue;
    const doc = JSON.parse(readFileSync(join(resultsDir, name), 'utf8'));
    const t = Number(String(doc.run_id).split('-').pop());
    if (!best || t > best.t) best = { t, name, doc };
  }
  return best;
}

// The numpy-check row of a pilot model from the pilot result files; a later file overrides an earlier one.
export function pilotForward(resultsDir, id) {
  let row = null;
  for (const f of ['k28-forward-pilot.json', 'k28-forward-k28.5.json', 'k28-forward-k28.6.json',
    'k28-forward-k28.6-step0.json']) {
    const p = join(resultsDir, f);
    if (!existsSync(p)) continue;
    for (const line of readFileSync(p, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      const r = JSON.parse(line);
      if (r.model === id) row = r;
    }
  }
  return row;
}

export function slimForward(row) {
  if (!row) return row;
  const { layer_state_max_diff: _layers, ...rest } = row;
  return rest;
}

export const slugOf = (id) => id.replace('/', '__');
