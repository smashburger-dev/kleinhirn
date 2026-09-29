// Official measurement suite (docs/PLAN.md gate runs): kleinhirn vs
// ONNX Runtime Web on small-upstream, kleinhirn WASM vs ORT-wasm, and
// kleinhirn Julia 1 vs the upstream WebGPU benchmark pages.
//
// Load gate: the suite starts only after the 1-minute load average stays
// below --max-load for GATE_MINUTES minutes (sampled every 60 s). Before
// each run it waits up to COOLDOWN_MS for the load to fall below --max-load
// again. A run is provisional only if the load at its start is above
// --max-load: the measured runtime's own threads raise the load during a
// run (ORT-web workers pushed it from 4.3 to 6.6), so load during the run is
// recorded but does not judge the run.
//
// Fairness: runs are grouped into comparison pairs (kleinhirn vs the
// measured baseline) and executed interleaved ABAB, REPS repetitions per
// pair. Each execution writes official-<name>-r<rep>.json; per run name
// an aggregate official-<name>.json reports median and range across the
// repetitions, with the 1-minute load of every execution.
//
// Usage: node bench/run-official.mjs [--no-wait] [--max-load N]
//              [--only name,...] [--reps N] [--tag T]
//   --no-wait    start immediately instead of waiting for the load gate
//                (a run is still provisional if its start load is too high)
//   --max-load   1-minute load ceiling for the gate and per-run marking
//                (default 6: below saturation on this 10-core M1 Pro)
//   --only       restrict the suite to the named runs or named pairs
//                (pair sides outside the set are skipped; pairs keep
//                their A/B interleave)
//   --reps       repetitions per pair (default 3)
//   --tag T      load-sensitivity mode: files go to
//                loadsens-T-<name>-r<rep>.json, no official-* aggregate
//                and no official-summary.json
//   --no-mem     no memory sampling (latency-only pass, peakMemMb null)
//   --mem-pass   dedicated memory pass: every selected run once (reps 1),
//                files mem-<name>.json, latency of this pass is not
//                reported. Memory sampling runs in a worker thread
//                (1 s interval) so the synchronous pgrep/footprint calls
//                cannot starve Playwright's event loop.
//   --prefix P   file and run_id prefix instead of official/mem/loadsens
//                (K20: k20-official, k20-mem-p1), so a pass does not
//                overwrite earlier official-* files. Per-run files are
//                numbered per run name across pairs (-r1..-rN); each file
//                records its pair and repetition.

import { chromium } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, cpSync, existsSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';
import { Worker } from 'node:worker_threads';
import { median, METRICS } from './mem.mjs';

const BASE = 'http://localhost:5199';
const JULIA_BASE = 'http://localhost:5198';
const NO_MEM = process.argv.includes('--no-mem');
const MB = 1024 * 1024;
const GATE_MINUTES = 10;
const COOLDOWN_MS = 5 * 60000;
const BUNDLE = 'dist/kleinhirn.js';
const RUNS_TSV = 'data/hillclimb/runs.tsv';
const noWait = process.argv.includes('--no-wait');
const argValue = (flag) => {
  const i = process.argv.indexOf(flag);
  return i >= 0 ? process.argv[i + 1] : null;
};
const MAX_LOAD = argValue('--max-load') !== null
  ? Number(argValue('--max-load')) : 6;
if (!Number.isFinite(MAX_LOAD)) {
  throw new Error(`--max-load needs a number, got ${argValue('--max-load')}`);
}
const MEM_PASS = process.argv.includes('--mem-pass');
if (MEM_PASS && NO_MEM) throw new Error('--mem-pass and --no-mem conflict');
const REPS = MEM_PASS
  ? 1
  : argValue('--reps') !== null ? Number(argValue('--reps')) : 3;
if (!Number.isInteger(REPS) || REPS < 1) {
  throw new Error(`--reps needs a positive integer, got ${argValue('--reps')}`);
}
const ONLY = argValue('--only') !== null
  ? new Set(argValue('--only').split(',')) : null;
const TAG = argValue('--tag') ?? (MEM_PASS ? 'mem' : null);
const PREFIX = argValue('--prefix')
  ?? (MEM_PASS ? 'mem' : TAG ? `loadsens-${TAG}` : 'official');

const GOLDEN = (L) => (L === 128 ? 'texts1000_l128k16' : `long200_l${L}k16`);

// name, page url, result global or 'status-json', kind for field mapping
const RUNS = [];
for (const precision of ['f16', 'f32']) {
  for (const L of [128, 256, 512, 1024]) {
    RUNS.push({
      name: `kh-small-${precision}-L${L}`,
      engine: 'kleinhirn', model: 'small-upstream', precision, bucket: L,
      url: `/bench/kleinhirn.html?model=small-upstream&precision=${precision}`
        + `&goldens=${GOLDEN(L)}&buckets=${L}&limits=minimum`,
      resultKey: 'khResult', timeoutMs: 900000,
    });
  }
}
for (const L of [128, 256]) {
  RUNS.push({
    name: `kh-wasm-small-L${L}`,
    engine: 'kleinhirn-wasm', model: 'small-upstream', precision: 'f32',
    bucket: L,
    url: `/bench/kleinhirn.html?model=small-upstream&precision=f32`
      + `&goldens=${GOLDEN(L)}&buckets=${L}&backend=wasm`,
    resultKey: 'khResult', timeoutMs: 1800000,
  });
}
for (const ep of ['webgpu', 'wasm']) {
  for (const precision of ['f16', 'f32']) {
    RUNS.push({
      name: `ort-small-${precision}-${ep}`,
      engine: 'ort-web', model: 'small-upstream', precision, ep,
      url: `/bench/ort.html?model=small-upstream&precision=${precision}&ep=${ep}`,
      resultKey: 'khOrtResult', timeoutMs: 600000,
    });
  }
}
// K20: the same ORT page on the offline-fused graph
// (convert/optimize_onnx.py). No wasm f16: that page hangs (FINDINGS §10).
for (const [ep, precision] of [['webgpu', 'f16'], ['webgpu', 'f32'], ['wasm', 'f32']]) {
  RUNS.push({
    name: `ort-small-${precision}-${ep}-opt`,
    engine: 'ort-web', model: 'small-upstream', precision, ep, graph: 'opt',
    url: `/bench/ort.html?model=small-upstream&precision=${precision}&ep=${ep}`
      + '&graph=opt',
    resultKey: 'khOrtResult', timeoutMs: 600000,
  });
}
// K20gc: ORT graph capture (enableGraphCapture, GPU-buffer IO), webgpu only.
for (const precision of ['f16', 'f32']) {
  for (const graph of ['std', 'opt', 'optgc']) {
    const label = graph === 'std' ? '' : `-${graph}`;
    for (const capture of graph === 'optgc' ? [false, true] : [true]) {
      RUNS.push({
        name: `ort-small-${precision}-webgpu${label}${capture ? '-capture' : ''}`,
        engine: 'ort-web', model: 'small-upstream', precision, ep: 'webgpu', graph,
        capture,
        url: `/bench/ort.html?model=small-upstream&precision=${precision}&ep=webgpu`
          + `&graph=${graph}${capture ? '&capture=1' : ''}`,
        resultKey: 'khOrtResult', timeoutMs: 600000,
      });
    }
  }
}
for (const precision of ['f16', 'f32']) {
  RUNS.push({
    name: `kh-julia-${precision}`,
    engine: 'kleinhirn-julia', model: 'julia-1', precision,
    url: `/bench/julia-bench.html?precision=${precision}&limits=minimum`,
    resultKey: 'khJuliaBench', timeoutMs: 600000,
  });
}
// The upstream julia pages are served by their own vite root: under the
// kleinhirn server (worktree with a symlinked models/) the nested relative
// symlink onnx/tokenizer.json -> ../repo/tokenizer/tokenizer.json returns
// 404, so the page fails in loadWasmEncoder before the benchmark starts.
// K16 batch matrix: time per decision inside a GPU batch at B=1,4,8,16.
// small-upstream runs the L128 golden set; julia-1 its 100 requests.
for (const precision of ['f16', 'f32']) {
  for (const B of [1, 4, 8, 16]) {
    RUNS.push({
      name: `kh-small-batch-${precision}-L128-B${B}`,
      engine: 'kleinhirn-batch', model: 'small-upstream', precision,
      bucket: 128, batch: B,
      url: `/bench/batch-bench.html?model=small-upstream&precision=${precision}`
        + `&bucket=128&batch=${B}&limits=minimum`,
      resultKey: 'khBatchBench', timeoutMs: 900000,
    });
  }
}
for (const precision of ['f16', 'f32']) {
  for (const B of [1, 4, 8, 16]) {
    RUNS.push({
      name: `kh-julia-batch-${precision}-B${B}`,
      engine: 'kleinhirn-julia-batch', model: 'julia-1', precision, batch: B,
      url: `/bench/batch-bench.html?model=julia-1&precision=${precision}`
        + `&batch=${B}&limits=minimum`,
      resultKey: 'khBatchBench', timeoutMs: 900000,
    });
  }
}
RUNS.push({
  name: 'upstream-julia-batch4',
  engine: 'julia-upstream', model: 'julia-1', batch: 4,
  base: JULIA_BASE, url: '/benchmark-webgpu.html',
  statusJson: true, click: '#run', timeoutMs: 900000,
});
RUNS.push({
  name: 'upstream-julia-batch1',
  engine: 'julia-upstream', model: 'julia-1', batch: 1,
  base: JULIA_BASE, url: '/bench-batch1.html',
  statusJson: true, click: '#run', timeoutMs: 900000,
});
// K20: upstream loop on a chosen graph file (bench/upstream-julia/
// bench-graph.html). "std" runs the unchanged upstream graph through the
// same page as a control for the page copy.
const JULIA_GRAPHS = {
  std: ['model.onnx', null],
  opt: ['model_opt.onnx', 'f32'],
  'opt-f16': ['model_opt_f16.onnx', 'f16'],
};
for (const [variant, [file, precision]] of Object.entries(JULIA_GRAPHS)) {
  for (const B of [4, 1]) {
    RUNS.push({
      name: `upstream-julia-batch${B}-graph-${variant}`,
      engine: 'julia-upstream', model: 'julia-1', batch: B,
      graph: variant, precision,
      base: JULIA_BASE, url: `/bench-graph.html?model=${file}&batch=${B}`,
      statusJson: true, click: '#run', timeoutMs: 900000,
    });
  }
}

// Comparison pairs: a = kleinhirn run, b = the baseline it is compared
// against (b may be null for coverage runs without a counterpart). Each
// pair is measured interleaved ABAB, REPS times: a, b, a, b, a, b.
const byName = Object.fromEntries(RUNS.map((r) => [r.name, r]));
const PAIRS = [];
for (const precision of ['f16', 'f32']) {
  for (const L of [128, 256, 512, 1024]) {
    PAIRS.push({
      name: `kh-vs-ort-webgpu small ${precision} L${L}`,
      a: byName[`kh-small-${precision}-L${L}`],
      b: byName[`ort-small-${precision}-webgpu`],
    });
  }
}
for (const L of [128, 256]) {
  PAIRS.push({
    name: `kh-wasm-vs-ort-wasm small f32 L${L}`,
    a: byName[`kh-wasm-small-L${L}`],
    b: byName['ort-small-f32-wasm'],
  });
}
// ort f16 on the wasm EP has no kleinhirn counterpart; run it for
// coverage in its own slot (a = null).
PAIRS.push({
  name: 'ort-wasm small f16 (no kleinhirn counterpart)',
  a: null,
  b: byName['ort-small-f16-wasm'],
});
for (const precision of ['f16', 'f32']) {
  for (const up of ['upstream-julia-batch4', 'upstream-julia-batch1']) {
    PAIRS.push({
      name: `kh-julia ${precision} vs ${up}`,
      a: byName[`kh-julia-${precision}`],
      b: byName[up],
    });
  }
}
// K16 batch matrix: each batch size interleaves against its own B1
// baseline, and the Julia B4 run pairs against the upstream batch-4
// page — the published comparison.
for (const family of ['kh-small-batch-f16-L128', 'kh-julia-batch-f16']) {
  for (const B of [4, 8, 16]) {
    PAIRS.push({
      name: `${family} B${B} vs B1`,
      a: byName[`${family}-B${B}`],
      b: byName[`${family}-B1`],
    });
  }
}
PAIRS.push({
  name: 'kh-julia-batch-f16-B4 vs upstream-julia-batch4',
  a: byName['kh-julia-batch-f16-B4'],
  b: byName['upstream-julia-batch4'],
});
// Published Julia pair: kleinhirn's fastest path (batch stride, B1)
// against upstream's fastest path (batch 4), measured interleaved.
PAIRS.push({
  name: 'kh-julia-batch-f16-B1 vs upstream-julia-batch4',
  a: byName['kh-julia-batch-f16-B1'],
  b: byName['upstream-julia-batch4'],
});
// K20: kleinhirn against the offline-fused ORT graphs (a = kleinhirn),
// and the plain ORT graph against the fused one (a = plain) to show what
// the optimizer changes, both interleaved ABAB.
for (const [kh, ort] of [
  ['kh-small-f16-L128', 'ort-small-f16-webgpu'],
  ['kh-small-f32-L128', 'ort-small-f32-webgpu'],
  ['kh-wasm-small-L128', 'ort-small-f32-wasm'],
]) {
  PAIRS.push({ name: `k20 ${kh} vs ${ort}-opt`, a: byName[kh], b: byName[`${ort}-opt`] });
  PAIRS.push({ name: `k20 ${ort} vs ${ort}-opt`, a: byName[ort], b: byName[`${ort}-opt`] });
}
for (const precision of ['f16', 'f32']) {
  const cap = byName[`ort-small-${precision}-webgpu-opt-capture`];
  PAIRS.push({
    name: `k20gc kh-small-${precision}-L128 vs ort-small-${precision}-webgpu-opt-capture`,
    a: byName[`kh-small-${precision}-L128`], b: cap,
  });
  PAIRS.push({
    name: `k20gc ort-small-${precision}-webgpu-opt vs ort-small-${precision}-webgpu-opt-capture`,
    a: byName[`ort-small-${precision}-webgpu-opt`], b: cap,
  });
}
for (const precision of ['f16', 'f32']) {
  const cap = byName[`ort-small-${precision}-webgpu-optgc-capture`];
  PAIRS.push({
    name: `k20gc kh-small-${precision}-L128 vs ort-small-${precision}-webgpu-optgc-capture`,
    a: byName[`kh-small-${precision}-L128`], b: cap,
  });
  PAIRS.push({
    name: `k20gc ort-small-${precision}-webgpu-opt vs ort-small-${precision}-webgpu-optgc-capture`,
    a: byName[`ort-small-${precision}-webgpu-opt`], b: cap,
  });
  PAIRS.push({
    name: `k20gc ort-small-${precision}-webgpu-optgc vs ort-small-${precision}-webgpu-optgc-capture`,
    a: byName[`ort-small-${precision}-webgpu-optgc`], b: cap,
  });
}
for (const B of [4, 1]) {
  for (const variant of Object.keys(JULIA_GRAPHS)) {
    PAIRS.push({
      name: `k20 kh-julia-batch-f16-B1 vs upstream-julia-batch${B}-graph-${variant}`,
      a: byName['kh-julia-batch-f16-B1'],
      b: byName[`upstream-julia-batch${B}-graph-${variant}`],
    });
  }
  PAIRS.push({
    name: `k20 upstream-julia-batch${B} vs graph-std (page control)`,
    a: byName[B === 4 ? 'upstream-julia-batch4' : 'upstream-julia-batch1'],
    b: byName[`upstream-julia-batch${B}-graph-std`],
  });
}
// Equal boundaries: kleinhirn decide() (tokenization included, one
// request per call) against the upstream batch-1 loop on the fused graph
// of the same precision.
for (const [precision, variant] of [['f16', 'opt-f16'], ['f32', 'opt']]) {
  PAIRS.push({
    name: `k20 kh-julia-${precision} vs upstream-julia-batch1-graph-${variant}`,
    a: byName[`kh-julia-${precision}`],
    b: byName[`upstream-julia-batch1-graph-${variant}`],
  });
}

if (ONLY) {
  const known = new Set([...PAIRS.map((p) => p.name), ...RUNS.map((r) => r.name)]);
  const unknown = [...ONLY].filter((n) => !known.has(n));
  if (unknown.length) throw new Error(`--only: unknown names: ${unknown.join(' | ')}`);
}

function gitCommit() {
  return execFileSync(
    'git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
}

// Protocol: every engine run gets a line in runs.tsv (pending, then
// updated after the run), same layout as the other runners.
function appendRun(runId, cfg) {
  const line = [
    new Date().toISOString().slice(0, 10), runId, 'official', gitCommit(),
    cfg.engine, cfg.model, cfg.bucket ? `L${cfg.bucket}K16` : 'julia',
    cfg.precision ?? '',
    '', '', '', '', '', '', '', '', '', '', '', '', '', 'pending',
  ].join('\t');
  appendFileSync(RUNS_TSV, line + '\n');
}

function finishRun(runId, fields) {
  const lines = readFileSync(RUNS_TSV, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  if (idx < 0) return;
  const cells = lines[idx].split('\t');
  const cols = [
    'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket', 'precision',
    'argmax_agreement', 'max_abs_logit_diff', 'max_abs_prob_diff', 'median_ms',
    'p95_ms', 'model_only_median_ms', 'load_ms', 'download_mb', 'gpu_mb',
    'peak_mem_mb', 'browser', 'adapter', 'kept', 'note',
  ];
  for (const [k, v] of Object.entries(fields)) cells[cols.indexOf(k)] = String(v);
  lines[idx] = cells.join('\t');
  writeFileSync(RUNS_TSV, lines.join('\n'));
}

function loadAvg1() {
  // loadavg() is POSIX getloadavg, portable across macOS and Linux; the
  // first value is the 1-minute average. NaN must never slip through: a
  // bad value throws instead of silently disabling the gate.
  const v = loadavg()[0];
  if (!Number.isFinite(v)) throw new Error(`cannot read loadavg: ${v}`);
  return v;
}

async function waitForLoadGate() {
  process.stdout.write(
    `load gate: waiting for 1-min load < ${MAX_LOAD} `
    + `for ${GATE_MINUTES} consecutive minutes\n`);
  let ok = 0;
  for (;;) {
    const l = loadAvg1();
    ok = l < MAX_LOAD ? ok + 1 : 0;
    process.stdout.write(
      `  ${new Date().toISOString().slice(11, 19)} load ${l} `
      + `(${ok}/${GATE_MINUTES})\n`);
    if (ok >= GATE_MINUTES) return;
    await new Promise((r) => setTimeout(r, 60000));
  }
}

const spawnedServers = [];
async function ensureServer(base = BASE, cwd = undefined) {
  try {
    await fetch(base, { signal: AbortSignal.timeout(3000) });
    return;
  } catch { /* not running */ }
  const port = new URL(base).port;
  spawnedServers.push(spawn(
    'npx', ['vite', '--port', port, '--strictPort'], { stdio: 'ignore', cwd }));
  for (let i = 0; i < 30; i += 1) {
    try {
      await fetch(base, { signal: AbortSignal.timeout(1000) });
      return;
    } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`could not start vite dev server on :${port}`);
}

function checkBundle() {
  if (!existsSync(BUNDLE)) {
    throw new Error(`missing ${BUNDLE}; run npm run build first`);
  }
  const buildId = readFileSync(BUNDLE, 'utf8')
    .match(/buildId\s*:\s*["']([a-z0-9]+)["']/)?.[1];
  return buildId;
}

async function checkServedBuild(buildId) {
  const served = await (await fetch(`${BASE}/dist/kleinhirn.js`)).text();
  if (!served.includes(buildId)) {
    throw new Error(
      `dev server serves a stale bundle (buildId ${buildId}); restart vite`);
  }
}

async function scrapeStatusJson(page, timeoutMs) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const text = await page.evaluate(
      () => document.querySelector('#status')?.textContent ?? '');
    const trimmed = text.trim();
    if (trimmed.startsWith('{')) {
      try { return JSON.parse(trimmed); } catch { /* partial */ }
    }
    if (/error|Error/.test(trimmed) && trimmed.length > 20) {
      throw new Error('page: ' + trimmed.slice(0, 500));
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error('timeout waiting for #status JSON');
}

async function runOne(cfg, browser, metricName) {
  const loadStart = loadAvg1();
  const page = await browser.newPage();
  const samples = [];
  let worker = null;
  let loadTimer = null;
  let loadExceeded = false;
  try {
    // Memory sampling lives in a worker thread: the synchronous
    // pgrep/footprint calls can take seconds on Julia's large process
    // tree and would starve Playwright's event loop on the main thread.
    if (!NO_MEM) {
      worker = new Worker(
        new URL('./mem-worker.mjs', import.meta.url),
        { workerData: {
          metric: metricName, rootPid: process.pid, intervalMs: 1000,
        } });
      worker.on('message', (s) => samples.push(s));
      worker.on('error', () => {});
    }
    loadTimer = setInterval(() => {
      try {
        if (loadAvg1() > MAX_LOAD) loadExceeded = true;
      } catch { /* transient */ }
    }, 10000);
    await new Promise((r) => setTimeout(r, 1500));
    const preGotoSamples = samples.length;
    await page.goto((cfg.base ?? BASE) + cfg.url);
    let raw;
    if (cfg.statusJson) {
      await page.click(cfg.click);
      raw = await scrapeStatusJson(page, cfg.timeoutMs);
    } else {
      const t0 = Date.now();
      while (Date.now() - t0 < cfg.timeoutMs) {
        const done = await page.evaluate(
          (k) => window[k]?.done ?? false, cfg.resultKey);
        if (done) break;
        await new Promise((r) => setTimeout(r, 500));
      }
      raw = await page.evaluate((k) => window[k], cfg.resultKey);
      if (raw?.error) throw new Error('page: ' + raw.error);
      if (!raw?.done) throw new Error('timeout waiting for page result');
    }
    const loadEnd = loadAvg1();
    const bytes = samples.map((s) => s.bytes).filter(Number.isFinite);
    const baseSamples = bytes.slice(0, Math.max(preGotoSamples, 1));
    const baselineMb = baseSamples.length ? median(baseSamples) / MB : 0;
    const peakMb = bytes.length ? Math.max(...bytes) / MB : baselineMb;
    const provisional = loadStart > MAX_LOAD;
    const out = {
      date: new Date().toISOString().slice(0, 10),
      run_id: `official-${cfg.name}`,
      official: !provisional,
      provisionalReason: provisional
        ? `1-min load ${loadStart} above ${MAX_LOAD} at run start` : null,
      engine: cfg.engine, model: cfg.model,
      precision: cfg.precision ?? null, bucket: cfg.bucket ?? null,
      backend: cfg.engine === 'kleinhirn-wasm' ? 'wasm' : 'webgpu',
      ep: cfg.ep ?? null, batchSize: cfg.batch ?? null,
      graph: cfg.graph ?? null, capture: cfg.capture ?? false,
      browser: `chromium-${browser.version()}`,
      adapterInfo: raw.adapterInfo ?? raw.info?.adapter ?? null,
      loadAvgStart: loadStart, loadAvgEnd: loadEnd,
      loadMaxObserved: Math.max(loadStart, loadEnd),
      baselineMb,
      peakMemMb: NO_MEM ? null : peakMb - baselineMb,
      page: raw,
    };
    return { out, loadExceeded: provisional && loadExceeded };
  } finally {
    if (loadTimer) clearInterval(loadTimer);
    if (worker) await worker.terminate();
    await page.close();
  }
}

function summarize(cfg, raw) {
  // Normalize the page result into the top-level comparison fields.
  const s = { latency: {}, parity: {} };
  if (cfg.engine === 'kleinhirn' || cfg.engine === 'kleinhirn-wasm') {
    s.latency = {
      medianMs: raw.endToEnd?.medianMs, p95Ms: raw.endToEnd?.p95Ms,
      modelOnlyMedianMs: raw.modelOnly?.medianMs,
      loadMs: raw.loadMs, loadMsWarm: raw.loadMsWarm,
    };
    s.downloadBytes = raw.info?.downloadBytes ?? null;
    s.gpuBytes = raw.info?.gpuBytes ?? null;
    s.parity = raw.parity ?? null;
    s.n = raw.n ?? null;
  } else if (cfg.engine === 'ort-web') {
    s.latency = {
      medianMs: raw.latency?.medianMs, p95Ms: raw.latency?.p95Ms,
      loadMs: raw.loadMs,
    };
    s.downloadBytes = raw.downloadBytes ?? null;
    s.onnxBytes = raw.onnxBytes ?? null;
    s.runtimeBytes = (raw.downloadBytes ?? 0) - (raw.onnxBytes ?? 0);
    s.parity = raw.parity ?? null;
    s.n = raw.n ?? null;
  } else if (cfg.engine === 'kleinhirn-batch'
    || cfg.engine === 'kleinhirn-julia-batch') {
    const r = raw.result ?? {};
    s.latency = {
      msPerDecision: r.ms_per_decision,
      batchMedianMs: r.batch_ms?.medianMs,
      batchP95Ms: r.batch_ms?.p95Ms,
      perDecisionMedianMs: r.per_decision?.medianMs,
      perDecisionP95Ms: r.per_decision?.p95Ms,
      loadMs: raw.loadMs,
    };
    s.downloadBytes = raw.info?.downloadBytes ?? null;
    s.gpuBytes = raw.info?.gpuBytes ?? null;
    s.parity = r.parity ?? null;
    s.n = r.n_requests ?? null;
    s.batchSize = r.batch_size ?? cfg.batch ?? null;
  } else if (cfg.engine === 'kleinhirn-julia') {
    const r = raw.result ?? {};
    s.latency = {
      medianMs: r.median_ms, msPerRequest: r.ms_per_request,
      perRequestMedianMs: r.per_request_median_ms,
      perRequestP95Ms: r.per_request_p95_ms,
      loadMs: r.load_ms, loadAndWarmMs: r.load_and_warm_ms,
    };
    s.downloadBytes = raw.info?.downloadBytes ?? null;
    s.gpuBytes = raw.info?.gpuBytes ?? null;
    s.parity = {
      matchingPredictions: r.matching_predictions,
      maxAbsLogitError: r.max_abs_logit_error_vs_pytorch,
    };
    s.n = r.requests ?? null;
  } else {
    // upstream pages: scrape fields as reported by their benchmark.
    s.latency = {
      medianMs: raw.median_ms, msPerRequest: raw.ms_per_request,
      loadAndWarmMs: raw.load_and_warm_ms,
    };
    s.parity = {
      matchingPredictions: raw.matching_predictions,
      maxAbsLogitError: raw.max_abs_logit_error_vs_pytorch,
    };
    s.n = raw.requests ?? null;
  }
  return s;
}

async function main() {
  if (!existsSync('bench/results/calibration-latest.json')) {
    throw new Error(
      'missing calibration-latest.json; run bench/run-calibrate.mjs first');
  }
  const calibration = JSON.parse(
    readFileSync('bench/results/calibration-latest.json', 'utf8'));
  if (!METRICS[calibration.chosen]) {
    throw new Error(`unknown calibration metric ${calibration.chosen}`);
  }
  const buildId = checkBundle();
  mkdirSync('bench/results', { recursive: true });

  // The K20 graph page is tracked in git; the upstream vite root lives
  // under the gitignored models/ tree.
  cpSync('bench/upstream-julia/bench-graph.html',
    'models/julia-1/onnx/bench-graph.html');
  await ensureServer();
  await ensureServer(JULIA_BASE, 'models/julia-1/onnx');
  await checkServedBuild(buildId);
  if (!noWait) await waitForLoadGate();

  const results = [];
  const byCfg = new Map(); // cfg.name -> per-repetition outcome records
  const record = (cfgName, entry) => {
    if (!byCfg.has(cfgName)) byCfg.set(cfgName, []);
    byCfg.get(cfgName).push(entry);
  };
  let aborted = false;

  // --only keeps only the named sides inside each pair; a named pair
  // keeps both sides. The A/B interleave per pair is unchanged.
  // --mem-pass flattens the suite: every selected run once, no
  // comparison pairing.
  const pairs = MEM_PASS
    ? RUNS.filter((r) => !ONLY || ONLY.has(r.name))
        .map((r) => ({ name: r.name, a: r, b: null }))
    : ONLY
    ? PAIRS.map((p) => ONLY.has(p.name) ? p : ({
      name: p.name,
      a: p.a && ONLY.has(p.a.name) ? p.a : null,
      b: p.b && ONLY.has(p.b.name) ? p.b : null,
    })).filter((p) => p.a || p.b)
    : PAIRS;

  const execCount = new Map(); // cfg.name -> executions so far
  for (const pair of pairs) {
    if (aborted) break;
    process.stdout.write(`pair ${pair.name}\n`);
    for (let rep = 1; rep <= REPS && !aborted; rep += 1) {
      for (const cfg of [pair.a, pair.b]) {
        if (!cfg) continue;
        if (aborted) break;
        for (const t0 = Date.now();
          loadAvg1() > MAX_LOAD && Date.now() - t0 < COOLDOWN_MS;) {
          await new Promise((r) => setTimeout(r, 15000));
        }
        process.stdout.write(`run ${cfg.name} (rep ${rep}) ...\n`);
        // Numbered per run name, so a run shared by two pairs keeps
        // every repetition on disk instead of the last pair's only.
        const seq = (execCount.get(cfg.name) ?? 0) + 1;
        execCount.set(cfg.name, seq);
        const runId = `${PREFIX}-${cfg.name}-r${seq}-${Date.now()}`;
        appendRun(runId, cfg);
        const browser = await chromium.launch(
          { headless: false, args: calibration.flags ?? [] });
        try {
          const { out } = await runOne(cfg, browser, calibration.chosen);
          out.run_id = runId;
          out.pair = pair.name;
          out.repetition = rep;
          Object.assign(out, summarize(cfg, out.page));
          const file = `bench/results/${PREFIX}-${cfg.name}-r${seq}.json`;
          out.file = file;
          writeFileSync(file, JSON.stringify(out, null, 1));
          results.push({
            name: cfg.name, rep, official: out.official, file });
          record(cfg.name, out);
          finishRun(runId, {
            median_ms: (out.latency?.msPerRequest
              ?? out.latency?.msPerDecision
              ?? out.latency?.medianMs)?.toFixed(2) ?? '',
            p95_ms: out.latency?.p95Ms?.toFixed(2) ?? '',
            model_only_median_ms: out.latency?.modelOnlyMedianMs?.toFixed(2)
              ?? '',
            load_ms: (out.latency?.loadMs ?? out.latency?.loadAndWarmMs ?? '')
              .toString().slice(0, 8),
            download_mb: ((out.downloadBytes ?? 0) / MB).toFixed(1),
            gpu_mb: out.gpuBytes ? (out.gpuBytes / MB).toFixed(0) : '',
            peak_mem_mb: out.peakMemMb?.toFixed(0) ?? '',
            browser: out.browser,
            adapter: JSON.stringify(out.adapterInfo).replace(/\t/g, ' '),
            kept: out.official ? 'official' : 'provisional',
            note: file,
          });
          process.stdout.write(
            `  -> ${file} official=${out.official}\n`);

        } catch (e) {
          results.push({ name: cfg.name, rep, error: String(e) });
          writeFileSync(
            `bench/results/${PREFIX}-${cfg.name}-r${seq}.json`,
            JSON.stringify({
              run_id: `${PREFIX}-${cfg.name}-r${seq}`, engine: cfg.engine,
              pair: pair.name, repetition: rep,
              official: false, error: String(e),
            }, null, 1));
          process.stdout.write(`  !! ${cfg.name} failed: ${e}\n`);
        } finally {
          await browser.close();
        }
      }
    }
  }

  // Per run name: aggregate the repetitions into <prefix>-<name>.json
  // with the median latency across reps and its range. In --tag mode the
  // aggregate uses the loadsens name too; official-*.json stays
  // untouched so a concurrent watcher cannot misread it.
  const aggregates = [];
  // Comparison metric per run kind: julia and batch runs report per-
  // request / per-decision times; medianMs on those runs is the total
  // over the request set and must not be compared against them.
  const perUnit = (o) => o.latency?.msPerRequest
    ?? o.latency?.msPerDecision ?? o.latency?.medianMs;
  const perUnitP95 = (o) => o.latency?.perRequestP95Ms
    ?? o.latency?.perDecisionP95Ms ?? o.latency?.p95Ms;
  for (const [name, outs] of byCfg) {
    const cfg = byName[name];
    const ok = outs.filter((o) => !o.error);
    const meds = ok.map(perUnit).filter(Number.isFinite);
    const p95s = ok.map(perUnitP95).filter(Number.isFinite);
    const agg = {
      run_id: `${PREFIX}-${name}`,
      engine: cfg.engine, model: cfg.model,
      precision: cfg.precision ?? null, bucket: cfg.bucket ?? null,
      ep: cfg.ep ?? null, batchSize: cfg.batch ?? null,
      graph: cfg.graph ?? null, capture: cfg.capture ?? false,
      backend: cfg.engine === 'kleinhirn-wasm' ? 'wasm' : 'webgpu',
      official: ok.length > 0 && ok.every((o) => o.official),
      repetitions: outs.length,
      medianMs: meds.length ? median(meds) : null,
      rangeMs: meds.length ? [Math.min(...meds), Math.max(...meds)] : null,
      p95Ms: p95s.length ? median(p95s) : null,
      downloadBytes: ok[0]?.downloadBytes ?? null,
      onnxBytes: ok[0]?.onnxBytes ?? null,
      runtimeBytes: ok[0]?.runtimeBytes ?? null,
      gpuBytes: ok[0]?.gpuBytes ?? null,
      peakMemMb: ok.length
        ? Math.max(...ok.map((o) => o.peakMemMb ?? 0)) : null,
      parity: ok.map((o) => o.parity ?? null),
      loadAvg: ok.map((o) => [o.loadAvgStart, o.loadAvgEnd]),
      repFiles: outs.map((o) => o.file),
      page: ok[0]?.page ?? null,
    };
    writeFileSync(`bench/results/${PREFIX}-${name}.json`,
      JSON.stringify(agg, null, 1));
    aggregates.push({ name, official: agg.official, medianMs: agg.medianMs });
  }

  writeFileSync(`bench/results/${PREFIX}-summary.json`, JSON.stringify({
    date: new Date().toISOString().slice(0, 10),
    maxLoad: MAX_LOAD, reps: REPS, tag: TAG,
    pairs: pairs.map((p) => p.name),
    runs: results, aggregates,
    aborted,
  }, null, 1));
  process.stdout.write('done\n');
}

try {
  await main();
} finally {
  for (const s of spawnedServers) s.kill();
}
