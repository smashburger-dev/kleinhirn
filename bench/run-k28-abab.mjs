// K28.1 gate G4/G5: official ABAB of the engine before K28 (A) against the
// K28.1 commit (B) on small-upstream L128 f16 and julia-1 f16, plus the
// bundle size. Started from the k28 worktree, it works in the official
// worktree (../kleinhirn-official): checkout --detach, npm run build, Vite on
// :5199, measure, stop Vite. Order A B A B A B A B A B, five repetitions per
// side, each measurement waits for the 1-minute load to fall below 6. The
// official worktree goes back to its original HEAD at the end, also on error.
// Usage: node bench/run-k28-abab.mjs [commitA=3bff7d6] [commitB=HEAD of k28] [--reps N]
// runs.tsv and result files land in the k28 worktree (KH_OUT_ROOT).

import { chromium } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync,
} from 'node:fs';
import { gzipSync } from 'node:zlib';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const K28 = resolve(fileURLToPath(new URL('..', import.meta.url)));
const OFFICIAL = resolve(K28, '../kleinhirn-official');
const PORT = 5199;
const BASE = `http://localhost:${PORT}`;
const MAX_LOAD = 6;
const LOAD_WAIT_MS = 30 * 60000;
const RUNS = resolve(K28, 'data/hillclimb/runs.tsv');
const RESULTS = resolve(K28, 'bench/results');

const args = process.argv.slice(2);
const repsIdx = args.indexOf('--reps');
const REPS = repsIdx >= 0 ? Number(args[repsIdx + 1]) : 5;
const pos = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--reps');
const commitA = pos[0] ?? '3bff7d6';
const commitB = pos[1] ?? execFileSync('git', ['-C', K28, 'rev-parse', '--short', 'HEAD'],
  { encoding: 'utf8' }).trim();

const sh = (cmd, a, opts = {}) => execFileSync(cmd, a, { encoding: 'utf8', ...opts }).trim();
const gitOff = (...a) => sh('git', ['-C', OFFICIAL, ...a]);
const load1 = () => loadavg()[0];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

async function waitForLoad() {
  const t0 = Date.now();
  while (load1() >= MAX_LOAD) {
    if (Date.now() - t0 > LOAD_WAIT_MS) throw new Error(`load stayed above ${MAX_LOAD} for 30 min`);
    await sleep(15000);
  }
}

function foreignBenchProcesses() {
  const out = [];
  for (const pat of ['vite', 'ms-playwright']) {
    try {
      const r = sh('pgrep', ['-fl', pat]);
      if (r) out.push(...r.split('\n').map((l) => l.slice(0, 120)));
    } catch { /* none */ }
  }
  return out;
}

const COLS = [
  'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket', 'precision',
  'argmax_agreement', 'max_abs_logit_diff', 'max_abs_prob_diff', 'median_ms',
  'p95_ms', 'model_only_median_ms', 'load_ms', 'download_mb', 'gpu_mb',
  'peak_mem_mb', 'browser', 'adapter', 'kept', 'note',
];

function appendRun(runId, change, commit) {
  const cells = COLS.map(() => '');
  Object.assign(cells, {
    0: new Date().toISOString().slice(0, 10), 1: runId, 2: change, 3: commit,
    4: 'kleinhirn-julia', 5: 'julia-1', 6: 'julia', 7: 'f16', 21: 'pending' });
  appendFileSync(RUNS, cells.join('\t') + '\n');
}

function finishRun(runId, fields) {
  const lines = readFileSync(RUNS, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  const cells = lines[idx].split('\t');
  for (const [k, v] of Object.entries(fields)) cells[COLS.indexOf(k)] = String(v);
  lines[idx] = cells.join('\t');
  writeFileSync(RUNS, lines.join('\n'));
}

async function startVite() {
  try {
    await fetch(BASE, { signal: AbortSignal.timeout(1500) });
    throw new Error(`port ${PORT} is taken`);
  } catch (e) {
    if (String(e).includes('taken')) throw e;
  }
  const proc = spawn('npx', ['vite', '--port', String(PORT), '--strictPort'], {
    cwd: OFFICIAL, stdio: 'ignore', detached: true });
  for (let i = 0; i < 40; i += 1) {
    try {
      await fetch(BASE, { signal: AbortSignal.timeout(1000) });
      return proc;
    } catch { await sleep(500); }
  }
  throw new Error('vite did not start');
}

async function stopVite(proc) {
  try { process.kill(-proc.pid, 'SIGTERM'); } catch { /* gone */ }
  for (let i = 0; i < 20; i += 1) {
    try {
      await fetch(BASE, { signal: AbortSignal.timeout(500) });
      await sleep(500);
    } catch { return; }
  }
  throw new Error(`vite on :${PORT} did not stop`);
}

function runSmall(change) {
  const out = execFileSync('node', [
    resolve(K28, 'bench/run-kleinhirn.mjs'), 'small-upstream', 'f16', change],
  { cwd: OFFICIAL, encoding: 'utf8', maxBuffer: 1 << 26,
    env: { ...process.env, KH_OUT_ROOT: K28 } });
  const r = JSON.parse(out.slice(out.indexOf('{')));
  return {
    e2e: r.endToEnd.medianMs, modelOnly: r.modelOnly.medianMs,
    loadAvg: r.loadAvg, file: r.run_id, argmax: r.parity.argmaxAgreement,
    gpuBytes: r.info?.gpuBytes ?? null,
  };
}

async function runJulia(change, commit) {
  const calibration = JSON.parse(
    readFileSync(resolve(RESULTS, 'calibration-latest.json'), 'utf8'));
  const runId = `kh-julia-f16-${change}-${Date.now()}`;
  appendRun(runId, change, commit);
  const browser = await chromium.launch({ headless: false, args: calibration.flags ?? [] });
  try {
    const page = await browser.newPage();
    await page.goto(`${BASE}/bench/julia-bench.html?precision=f16&limits=minimum`);
    const t0 = Date.now();
    while (Date.now() - t0 < 600000) {
      if (await page.evaluate(() => window.khJuliaBench?.done ?? false)) break;
      await sleep(500);
    }
    const raw = await page.evaluate(() => window.khJuliaBench);
    if (raw?.error) throw new Error(`page: ${raw.error}`);
    if (!raw?.done) throw new Error('julia bench timeout');
    const r = raw.result;
    const file = resolve(RESULTS, `${new Date().toISOString().slice(0, 10)}-${runId}.json`);
    writeFileSync(file, JSON.stringify({
      run_id: runId, change, commit, browser: `chromium-${browser.version()}`,
      loadAvg: loadavg().map((v) => v.toFixed(2)).join(' '), page: raw }, null, 1));
    finishRun(runId, {
      argmax_agreement: (r.matching_predictions / r.requests).toFixed(4),
      max_abs_logit_diff: r.max_abs_logit_error_vs_pytorch.toExponential(2),
      median_ms: r.ms_per_request.toFixed(2),
      p95_ms: r.per_request_p95_ms.toFixed(2),
      load_ms: `${r.load_ms.toFixed(0)}/${r.load_and_warm_ms.toFixed(0)}warm`,
      gpu_mb: ((raw.info?.gpuBytes ?? 0) / 1048576).toFixed(0),
      browser: `chromium-${browser.version()}`,
      adapter: JSON.stringify(raw.info?.adapter ?? {}).replace(/\t/g, ' '),
      kept: '', note: file });
    return {
      e2e: r.ms_per_request, perRequestMedian: r.per_request_median_ms,
      modelOnly: null, file: runId, argmax: r.matching_predictions / r.requests,
      gpuBytes: raw.info?.gpuBytes ?? null,
    };
  } finally {
    await browser.close();
  }
}

const bundleSizes = () => {
  const p = resolve(OFFICIAL, 'dist/kleinhirn.js');
  const raw = statSync(p).size;
  return { raw, gzip9: gzipSync(readFileSync(p), { level: 9 }).byteLength };
};

const summary = {
  commitA, commitB, reps: REPS, order: 'ABAB', maxLoad: MAX_LOAD,
  small: { A: [], B: [] }, julia: { A: [], B: [] }, bundle: {},
};

async function main() {
  const dirty = gitOff('status', '--short');
  if (dirty) throw new Error(`official worktree is not clean:\n${dirty}`);
  const others = foreignBenchProcesses();
  if (others.length) throw new Error(`other bench processes running:\n${others.join('\n')}`);
  const original = gitOff('rev-parse', 'HEAD');
  console.log(`official HEAD ${original.slice(0, 7)}, A=${commitA} B=${commitB}, ${REPS} reps`);
  mkdirSync(RESULTS, { recursive: true });
  try {
    for (let rep = 1; rep <= REPS; rep += 1) {
      for (const [side, commit] of [['A', commitA], ['B', commitB]]) {
        gitOff('checkout', '--detach', commit);
        execFileSync('npm', ['run', 'build'], { cwd: OFFICIAL, stdio: 'ignore' });
        summary.bundle[side] = bundleSizes();
        const change = `k28.1-abab-${side}`;
        const vite = await startVite();
        try {
          await waitForLoad();
          const s = runSmall(change);
          summary.small[side].push({ rep, commit, ...s });
          console.log(`rep ${rep} ${side} small e2e ${s.e2e.toFixed(3)} model ${s.modelOnly.toFixed(3)} load ${s.loadAvg}`);
          await waitForLoad();
          const loadBefore = load1();
          const j = await runJulia(change, commit);
          summary.julia[side].push({ rep, commit, loadStart: loadBefore, ...j });
          console.log(`rep ${rep} ${side} julia ${j.e2e.toFixed(4)} ms/request (start load ${loadBefore.toFixed(2)})`);
        } finally {
          await stopVite(vite);
        }
      }
    }
  } finally {
    gitOff('checkout', '--detach', original);
    console.log(`official worktree back on ${gitOff('rev-parse', '--short', 'HEAD')}`);
  }
}

function digest(list, key) {
  const values = list.map((x) => x[key]).filter((v) => v !== null);
  if (!values.length) return null;
  return { values, median: median(values), min: Math.min(...values), max: Math.max(...values),
    spreadPct: (Math.max(...values) / Math.min(...values) - 1) * 100 };
}

try {
  await main();
} finally {
  const out = { ...summary };
  for (const m of ['small', 'julia']) {
    out[`${m}Stats`] = {};
    for (const side of ['A', 'B']) {
      out[`${m}Stats`][side] = {
        e2eMs: digest(summary[m][side], 'e2e'),
        modelOnlyMs: digest(summary[m][side], 'modelOnly'),
        gpuBytes: summary[m][side].map((x) => x.gpuBytes),
        startLoads: summary[m][side].map((x) => x.loadAvg ?? x.loadStart),
      };
    }
    const a = out[`${m}Stats`].A.e2eMs;
    const b = out[`${m}Stats`].B.e2eMs;
    if (a && b) {
      out[`${m}Stats`].ratioE2E = b.median / a.median;
      out[`${m}Stats`].g4Pass = b.median / a.median <= 1.02;
      out[`${m}Stats`].aSpreadAbove2Pct = a.spreadPct > 2;
      const am = out[`${m}Stats`].A.modelOnlyMs;
      const bm = out[`${m}Stats`].B.modelOnlyMs;
      if (am && bm) out[`${m}Stats`].ratioModelOnly = bm.median / am.median;
    }
  }
  if (out.bundle.A && out.bundle.B) {
    out.bundle.deltaRaw = out.bundle.B.raw - out.bundle.A.raw;
    out.bundle.deltaGzip9 = out.bundle.B.gzip9 - out.bundle.A.gzip9;
  }
  const file = resolve(RESULTS, 'k28.1-abab-summary.json');
  writeFileSync(file, JSON.stringify(out, null, 1));
  console.log(`summary ${file}`);
}
