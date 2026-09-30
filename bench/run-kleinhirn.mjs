// kleinhirn engine benchmark runner: visible Playwright Chromium on
// bench/kleinhirn.html (imports the dist bundle built by `vite build`),
// memory sampled with the calibrated footprint metric. A pending runs.tsv
// line is appended before the run and updated afterwards.
// Usage: node bench/run-kleinhirn.mjs <model> <precision> [change]
// KH_OUT_ROOT=<dir>: runs.tsv, calibration and result files live under <dir>
// (official runs start from the official worktree and write into main).

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync,
} from 'node:fs';
import { gzipSync } from 'node:zlib';
import { resolve } from 'node:path';
import { median, METRICS, ownTreePids } from './mem.mjs';

const BASE = 'http://localhost:5199';
const MB = 1024 * 1024;
const OUT = resolve(process.env.KH_OUT_ROOT ?? '.');
const RUNS = resolve(OUT, 'data/hillclimb/runs.tsv');
const RESULTS = resolve(OUT, 'bench/results');
const BUNDLE = 'dist/kleinhirn.js';

const model = process.argv[2] ?? 'small-upstream';
const precision = process.argv[3] ?? 'f16';
const change = process.argv[4] ?? 'baseline';
const buckets = process.argv[5] ?? '';
const limits = process.argv[6] ?? 'minimum';
const backend = process.argv[7] ?? '';
const goldens = process.argv[8] ?? '';
const profile = process.argv[9] ?? '';

function gitCommit() {
  return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
}

function appendRun(runId) {
  const line = [
    new Date().toISOString().slice(0, 10), runId, change, gitCommit(),
    'kleinhirn', model, 'L128K16', precision,
    '', '', '', '', '', '', '', '', '', '', '', '', '', 'pending',
  ].join('\t');
  appendFileSync(RUNS, line + '\n');
}

function finishRun(runId, fields) {
  const lines = readFileSync(RUNS, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  const cells = lines[idx].split('\t');
  const cols = [
    'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket', 'precision',
    'argmax_agreement', 'max_abs_logit_diff', 'max_abs_prob_diff', 'median_ms',
    'p95_ms', 'model_only_median_ms', 'load_ms', 'download_mb', 'gpu_mb',
    'peak_mem_mb', 'browser', 'adapter', 'kept', 'note',
  ];
  for (const [k, v] of Object.entries(fields)) cells[cols.indexOf(k)] = String(v);
  lines[idx] = cells.join('\t');
  writeFileSync(RUNS, lines.join('\n'));
}

async function main() {
  if (!existsSync(resolve(RESULTS, 'calibration-latest.json'))) {
    throw new Error('missing calibration-latest.json; run bench/run-calibrate.mjs first');
  }
  if (!existsSync(BUNDLE)) {
    throw new Error(`missing ${BUNDLE}; run npx vite build first`);
  }
  // The vite dev server can serve a stale transform of dist/ after a rebuild.
  // The bundle carries a build id (__KH_BUILD_ID__ define); the value served
  // over HTTP must match the file on disk before any number is trusted.
  const diskBundle = readFileSync(BUNDLE, 'utf8');
  const buildId = diskBundle.match(/buildId\s*:\s*["']([a-z0-9]+)["']/)?.[1];
  if (!buildId) {
    throw new Error('no buildId in bundle; rebuild after vite.config define was added');
  }
  const served = await (await fetch(`${BASE}/dist/kleinhirn.js`)).text();
  if (!served.includes(buildId)) {
    throw new Error(
      `dev server serves a stale bundle (served lacks buildId ${buildId});`
      + ' restart the vite dev server');
  }
  const calibration = JSON.parse(readFileSync(resolve(RESULTS, 'calibration-latest.json'), 'utf8'));
  const metricFn = METRICS[calibration.chosen];
  const bundleRaw = statSync(BUNDLE).size;
  const bundleGzip = gzipSync(readFileSync(BUNDLE)).byteLength;
  const runId = `kh-${model}-${precision}${buckets ? `-L${buckets}` : ''}-${Date.now()}`;
  appendRun(runId);
  mkdirSync(RESULTS, { recursive: true });

  const browser = await chromium.launch({ headless: false, args: [
    ...(calibration.flags ?? []),
    ...(process.env.CHROMIUM_ARGS ?? '').split(' ').filter(Boolean),
  ] });
  const samples = [];
  let timer = null;
  let result;
  try {
    const page = await browser.newPage();
    await new Promise((r) => setTimeout(r, 1500));
    const tick = async () => {
      try {
        const pids = ownTreePids();
        samples.push({ t: Date.now(), pids: pids.length, bytes: metricFn(pids) });
      } catch { /* process tree mid-change */ }
    };
    timer = setInterval(tick, 100);
    const loadAvg = loadavg().map((v) => v.toFixed(2)).join(' ');
    await page.goto(
      `${BASE}/bench/kleinhirn.html?model=${model}&precision=${precision}`
      + `&limits=${limits}${buckets ? `&buckets=${buckets}` : ''}`
      + `${backend ? `&backend=${backend}` : ''}`
      + `${goldens ? `&goldens=${goldens}` : ''}`
      + `${profile ? `&profile=${profile}` : ''}`);
    const t0 = Date.now();
    // wasm runs 2x1000 sequential forwards at ~0.1-0.2s each; budget for it.
    while (Date.now() - t0 < (backend === 'wasm' ? 1800000 : 900000)) {
      const done = await page.evaluate(() => window.khResult?.done ?? false);
      if (done) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    await tick();
    clearInterval(timer);
    result = await page.evaluate(() => window.khResult);
    if (result?.error) throw new Error('page: ' + result.error);
    const adapterInfo = result?.adapterInfo ?? null;
    if (!adapterInfo && backend !== 'wasm') {
      throw new Error('no WebGPU adapter info');
    }
    const baselineMb = median(samples.slice(0, 10).map((s) => s.bytes)) / MB;
    const peakMb = Math.max(...samples.map((s) => s.bytes)) / MB;
    const out = {
      date: new Date().toISOString().slice(0, 10),
      run_id: runId,
      engine: 'kleinhirn',
      model,
      precision,
      bucket: goldens || 'L128K16',
      backend: backend || 'webgpu',
      browser: `chromium-${browser.version()}`,
      flags: calibration.flags ?? [],
      loadAvg, // 1-min load; official runs need it <= 4 at run start
      adapterInfo,
      memMetric: calibration.chosen,
      baselineMb,
      peakMemMb: peakMb - baselineMb,
      loadMs: result.loadMs,
      loadMsWarm: result.loadMsWarm,
      info: result.info,
      bundleBytes: bundleRaw,
      bundleGzipBytes: bundleGzip,
      endToEnd: result.endToEnd,
      modelOnly: result.modelOnly,
      profile: result.profile,
      parity: result.parity,
      n: result.n,
      sampleCount: samples.length,
      sampleIntervalMs: Number(median(
        samples.slice(1).map((s, i) => s.t - samples[i].t)).toFixed(0)),
    };
    const file = `${RESULTS}/${out.date}-${runId}.json`;
    writeFileSync(file, JSON.stringify(out, null, 1));
    finishRun(runId, {
      argmax_agreement: out.parity.argmaxAgreement.toFixed(4),
      max_abs_logit_diff: out.parity.maxAbsLogitDiff.toExponential(2),
      max_abs_prob_diff: out.parity.maxAbsProbDiff.toExponential(2),
      median_ms: out.endToEnd.medianMs.toFixed(2),
      p95_ms: out.endToEnd.p95Ms.toFixed(2),
      model_only_median_ms: out.modelOnly.medianMs.toFixed(2),
      load_ms: `${out.loadMs.toFixed(0)}/${out.loadMsWarm.toFixed(0)}warm`,
      download_mb: ((out.info?.downloadBytes ?? 0) / MB).toFixed(1),
      gpu_mb: ((out.info?.gpuBytes ?? 0) / MB).toFixed(0),
      peak_mem_mb: out.peakMemMb.toFixed(0),
      browser: out.browser,
      adapter: JSON.stringify(adapterInfo).replace(/\t/g, ' '),
      kept: '',
      note: file,
    });
    console.log(JSON.stringify(out, null, 1));
  } finally {
    if (timer) clearInterval(timer);
    await browser.close();
  }
}

await main();
