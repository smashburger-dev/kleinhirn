// Builds bench/results/k5-matrix-official.json from the per-rep files of the
// official K5 device matrix (bench/k5-orchestrate.mjs): k5-official-<row>-<stamp>-r<n>.json
// plus the control pair (k5-control-page-*, and the k5-control-runner lines in runs.tsv).
//
// Usage: node bench/k5-matrix.mjs [--out-root <dir>] [--notes <json file>]
//   --notes  extra records copied verbatim into the summary under "notes"
//            (e.g. the Firefox wasm per-item time from the diagnosis).
//
// Medians across reps are computed per number (median of the reps' medians,
// median of the reps' p95). Spread is min and max across reps.
//
// Gates (PLAN K5): parity per stage as K3, checked by tools/check_device_result.mjs
// on every rep file; L256 p95: WebGPU stages <= 20 ms, wasm on the M1 Pro <= 300 ms.
// The p95 gate is judged on the end-to-end p95 (the number an app sees) and the
// model-only p95 is reported next to it.

import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { checkResult } from '../tools/check_device_result.mjs';

const arg = (name, dflt = null) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const OUT = resolve(arg('out-root', '.'));
const RESULTS = resolve(OUT, 'bench/results');
const STAGES = ['f16', 'f32', 'wasm'];
const BUCKETS = ['L128', 'L256'];
const P95_GATE_MS = { f16: 20, f32: 20, wasm: 300 };
const ROW_ORDER = ['chromium', 'brave', 'webkit', 'safari', 'firefox', 'chromium-nogpu'];

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  if (!s.length) return null;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const spread = (xs) => (xs.length ? { min: Math.min(...xs), max: Math.max(...xs) } : null);
const stat = (xs) => (xs.length ? { median: median(xs), ...spread(xs), n: xs.length } : null);
const readJson = (f) => JSON.parse(readFileSync(resolve(RESULTS, f), 'utf8'));

// ---- matrix files -----------------------------------------------------------
const FILE = /^k5-official-(.+?)-(\d{8}T\d+)-r(\d+)\.json$/;
const byRow = new Map();
for (const f of readdirSync(RESULTS).sort()) {
  const m = FILE.exec(f);
  if (!m) continue;
  const w = readJson(f);
  if (!byRow.has(m[1])) byRow.set(m[1], []);
  byRow.get(m[1]).push({ file: f, rep: Number(m[3]), ...w });
}

function repSummary({ file, run, result }) {
  const base = {
    rep: run.rep, file, runId: run.runId, browser: run.browser, loadAvgStart: run.loadAvgStart, loadAvgEnd: run.loadAvgEnd,
    waitedMin: run.waitedMin, provisional: run.provisional, commit: run.commit, dirtyTree: run.dirtyTree,
    weights: run.weights, safariMode: run.safariMode ?? null, memory: run.memory,
    // frontmost app at each poll, entries when it changed; more than one app
    // means the browser window may have been in the background (K5 diagnosis c).
    frontApps: [...new Set((run.frontLog ?? []).map((x) => x.front))],
    frontChanges: Math.max(0, (run.frontLog ?? []).length - 1),
  };
  if (!result || result.error) return { ...base, error: result?.error ?? 'no result', stages: {} };
  const resultFile = file.replace(/\.json$/, '.result.json');
  const checked = checkResult(readJson(resultFile));
  const stages = {};
  for (const st of result.stages) {
    const v = checked.verdicts.find((x) => x.name === st.name);
    if (!st.ok) { stages[st.name] = { ok: false, error: st.error, errorPhase: st.errorPhase }; continue; }
    const lat = {};
    for (const b of BUCKETS) {
      lat[b] = {};
      for (const kind of ['modelOnly', 'e2e']) {
        const l = st.latency[b][kind];
        lat[b][kind] = { medianMs: l.medianMs, p95Ms: l.p95Ms, n: l.n, skipped: l.skipped };
      }
    }
    const par = {};
    for (const b of BUCKETS) {
      const s = st.parity[b].summary;
      par[b] = { argmaxAgreement: s.argmaxAgreement, maxAbsLogitDiff: s.maxAbsLogitDiff, maxAbsProbDiff: s.maxAbsProbDiff, missing: st.parity[b].missing };
    }
    stages[st.name] = {
      ok: true, parityPass: st.parityPass, checkerPass: v?.pass ?? false, latency: lat, parity: par,
      loadWallMs: st.load.wallMs, gpuBytes: st.memory.gpuBytes,
      jsHeapPeakMinusBaselineBytes: st.memory.jsHeapMeasurable ? st.memory.jsHeapPeakBytes - st.memory.jsHeapBaselineBytes : null,
    };
  }
  return {
    ...base, checkerErrors: checked.errors, protocol: result.protocol ?? null,
    environment: {
      userAgent: result.environment.userAgent, hardwareConcurrency: result.environment.hardwareConcurrency,
      crossOriginIsolated: result.environment.crossOriginIsolated ?? null, wasmSimd: result.environment.wasmSimd ?? null,
    },
    adapter: result.environment.webgpu?.adapter?.info ?? null, autoStage: result.autoStage?.picked ?? null, stages,
  };
}

function stageAcrossReps(reps, name) {
  const ran = reps.map((r) => r.stages[name]).filter(Boolean);
  const oks = ran.filter((s) => s.ok);
  if (!ran.length) return null;
  if (!oks.length) return { ran: false, repsTried: ran.length, errors: [...new Set(ran.map((s) => s.error))] };
  const out = { ran: true, repsOk: oks.length, repsTried: ran.length };
  out.parity = {
    passAllReps: oks.length === ran.length && oks.every((s) => s.parityPass && s.checkerPass),
    minArgmaxAgreement: Math.min(...oks.flatMap((s) => BUCKETS.map((b) => s.parity[b].argmaxAgreement))),
    maxAbsLogitDiff: Math.max(...oks.flatMap((s) => BUCKETS.map((b) => s.parity[b].maxAbsLogitDiff))),
    maxAbsProbDiff: Math.max(...oks.flatMap((s) => BUCKETS.map((b) => s.parity[b].maxAbsProbDiff))),
  };
  out.latency = {};
  for (const b of BUCKETS) {
    out.latency[b] = {};
    for (const kind of ['modelOnly', 'e2e']) {
      out.latency[b][kind] = {
        median: stat(oks.map((s) => s.latency[b][kind].medianMs)),
        p95: stat(oks.map((s) => s.latency[b][kind].p95Ms)),
      };
    }
  }
  out.loadWallMs = stat(oks.map((s) => s.loadWallMs));
  out.gpuBytes = oks[0].gpuBytes;
  out.jsHeapPeakMinusBaselineMb = stat(oks.map((s) => s.jsHeapPeakMinusBaselineBytes).filter((v) => v !== null).map((v) => v / 1048576));
  const gate = P95_GATE_MS[name];
  const p95e2e = out.latency.L256.e2e.p95.median;
  const p95mo = out.latency.L256.modelOnly.p95.median;
  out.gates = {
    parity: out.parity.passAllReps ? 'pass' : 'fail',
    l256P95: { limitMs: gate, e2eMs: p95e2e, modelOnlyMs: p95mo, e2e: p95e2e <= gate ? 'pass' : 'fail', modelOnly: p95mo <= gate ? 'pass' : 'fail' },
  };
  return out;
}

const rows = [];
for (const id of [...ROW_ORDER, ...[...byRow.keys()].filter((k) => !ROW_ORDER.includes(k))]) {
  const files = byRow.get(id);
  if (!files) continue;
  const reps = files.sort((a, b) => a.rep - b.rep).map(repSummary);
  const stageNames = STAGES.filter((s) => reps.some((r) => r.stages[s]));
  const memMb = reps.map((r) => r.memory?.peakMinusBaselineMb).filter((v) => typeof v === 'number');
  rows.push({
    row: id,
    browser: reps[0].browser,
    reps: reps.map((r) => ({
      rep: r.rep, file: r.file, loadAvgStart: r.loadAvgStart, waitedMin: r.waitedMin, provisional: r.provisional,
      dirtyTree: r.dirtyTree, frontApps: r.frontApps, frontChanges: r.frontChanges, error: r.error ?? null, checkerErrors: r.checkerErrors ?? null, safariMode: r.safariMode,
      memoryPeakMinusBaselineMb: r.memory?.peakMinusBaselineMb ?? null,
    })),
    runMemoryPeakMinusBaselineMb: stat(memMb),
    runMemoryMethod: reps[0].memory?.method ?? null,
    stages: Object.fromEntries(stageNames.map((s) => [s, stageAcrossReps(reps, s)])),
    adapter: reps.find((r) => r.adapter)?.adapter ?? null,
    environment: reps.find((r) => r.environment)?.environment ?? null,
  });
}

// ---- control pair -------------------------------------------------------------
function control() {
  const page = readdirSync(RESULTS).filter((f) => /^k5-control-page-chromium-.*-r\d+\.json$/.test(f)).sort().map((f) => {
    const { run, result } = readJson(f);
    const s = result.stages.find((x) => x.name === 'f16');
    const l = s.latency.L128;
    return {
      file: f, rep: run.rep, loadAvgStart: run.loadAvgStart,
      modelOnly: { medianMs: l.modelOnly.medianMs, p95Ms: l.modelOnly.p95Ms },
      e2e: { medianMs: l.e2e.medianMs, p95Ms: l.e2e.p95Ms },
    };
  });
  const tsv = readFileSync(resolve(OUT, 'data/hillclimb/runs.tsv'), 'utf8').split('\n').map((l) => l.split('\t'));
  const runner = tsv.filter((c) => c[2] === 'k5-control-runner').map((c) => {
    const file = c[21];
    const j = JSON.parse(readFileSync(file, 'utf8'));
    return {
      file: file.replace(`${OUT}/`, ''), runId: c[1], n: j.n, loadAvg: j.loadAvg,
      modelOnly: { medianMs: j.modelOnly.medianMs, p95Ms: j.modelOnly.p95Ms },
      e2e: { medianMs: j.endToEnd.medianMs, p95Ms: j.endToEnd.p95Ms },
    };
  });
  const agg = (xs, kind) => ({
    median: stat(xs.map((x) => x[kind].medianMs)),
    p95: stat(xs.map((x) => x[kind].p95Ms)),
  });
  return {
    description: 'ABAB, L128 f16, Chromium: site page (stages=f16) against bench/run-kleinhirn.mjs small-upstream f16 (968 goldens). FINDINGS section 16 reference: kleinhirn f16 6.5 ms median, p95 11.5.',
    page: { reps: page, modelOnly: agg(page, 'modelOnly'), e2e: agg(page, 'e2e') },
    runner: { reps: runner, modelOnly: agg(runner, 'modelOnly'), e2e: agg(runner, 'e2e') },
  };
}

// Firefox wasm was not run as a full stage: the per-item times of the
// diagnosis (limit 20, warm-up 5, trace) give the projected wall time of a full stage.
function firefoxWasmDiag() {
  const files = readdirSync(RESULTS).filter((f) => /^k5-diag-firefox-wasm(-front)?-firefox-\d+T\d+\.json$/.test(f)).sort();
  const runs = files.map((f) => {
    const { run, result } = readJson(f);
    const st = result.stages.find((x) => x.name === 'wasm');
    const per = {};
    for (const b of BUCKETS) {
      const mo = st.trace[b].modelOnlyMs;
      const e2e = st.trace[b].e2eMs;
      per[b] = { modelOnlyMedianMs: median(mo), modelOnlyMinMs: Math.min(...mo), modelOnlyMaxMs: Math.max(...mo), e2eMedianMs: median(e2e), items: mo.length };
    }
    // Full stage: per bucket 20 warm-up items x 2 calls, then 200 items in each of the two timed loops.
    const projectedMin = BUCKETS.reduce((sum, b) => sum + (40 + 400) * per[b].e2eMedianMs, 0) / 60000;
    return {
      file: f, browser: run.browser, protocol: result.protocol, crossOriginIsolated: result.environment.crossOriginIsolated,
      wasmSimd: result.environment.wasmSimd, hardwareConcurrency: result.environment.hardwareConcurrency,
      workerHardwareConcurrency: result.environment.workerHardwareConcurrency, frontApps: [...new Set((run.frontLog ?? []).map((x) => x.front))],
      perItem: per, projectedFullStageMinutes: Number(projectedMin.toFixed(0)),
    };
  });
  return { note: 'Full stage not run: projected wall time above 20 min. Per-item times come from the diagnosis runs (uniformly slow, no stalls).', runs };
}

const orchestrationState = (() => {
  try { return JSON.parse(readFileSync(resolve(RESULTS, 'k5-official-state.json'), 'utf8')); } catch { return null; }
})();
const commits = [...new Set(rows.flatMap((r) => r.reps.map((x) => x.file)).map((f) => f && readJson(f).run.commit))];
const summary = {
  schema: 'kleinhirn-k5-matrix-official/1',
  generatedBy: 'bench/k5-matrix.mjs',
  generatedAt: new Date().toISOString(),
  commit: commits,
  protocol: 'warm-up 20, all 200 items per bucket, 3 reps, fresh browser per rep, interleaved across rows (Safari reps ran after the other rows); weights=local (same files as HF revision f664c7a5ad15c81f1258e72c8f90925b64711d1d, sha256 identical)',
  loadRule: 'start-load rule < 6 (Noa, 30.09.)',
  loadRuleNote: 'The runs were started under the stricter limit of 4 that applied before Noa raised it to 6; maxStartLoad1min below shows the highest 1-minute load at any start.',
  maxStartLoad1min: Math.max(...rows.flatMap((r) => r.reps.map((x) => Number(String(x.loadAvgStart).split(' ')[0])))),
  gates: { parity: 'per stage as K3 (f32 and wasm exact, f16 >= 99.5 % argmax), all reps', l256P95Ms: P95_GATE_MS, judgedOn: 'median across reps of the end-to-end p95; model-only shown too' },
  browserVersions: Object.fromEntries(rows.map((r) => [r.row, r.browser])),
  loadAtStart: Object.fromEntries(rows.map((r) => [r.row, r.reps.map((x) => x.loadAvgStart)])),
  loadWaitMinutesTotal: orchestrationState ? Math.round(orchestrationState.waitMs / 60000) : null,
  rows,
  control: control(),
  firefoxWasm: firefoxWasmDiag(),
  notes: arg('notes') ? JSON.parse(readFileSync(arg('notes'), 'utf8')) : null,
};
writeFileSync(resolve(RESULTS, 'k5-matrix-official.json'), JSON.stringify(summary, null, 1));
console.log(`wrote ${resolve(RESULTS, 'k5-matrix-official.json')}: ${rows.length} rows`);
for (const r of rows) {
  for (const [name, s] of Object.entries(r.stages)) {
    if (!s?.ran) { console.log(`${r.row} ${name}: did not run`); continue; }
    const g = s.gates;
    console.log(`${r.row} ${name}: parity ${g.parity}, L256 e2e p95 ${g.l256P95.e2eMs.toFixed(1)} (${g.l256P95.e2e}), model-only p95 ${g.l256P95.modelOnlyMs.toFixed(1)}`);
  }
}
