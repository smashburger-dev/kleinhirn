// ORT WebGPU baseline runner: visible Playwright Chromium on bench/ort.html,
// memory sampled every 100 ms with the calibrated metric (calibration run
// must have written bench/results/calibration-latest.json). Before the run a
// pending line is appended to data/hillclimb/runs.tsv and updated in place
// afterwards. Result JSON: bench/results/<date>-ort-<model>-<precision>.json.

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';
import { median, METRICS, ownTreePids } from './mem.mjs';

const BASE = 'http://localhost:5199';
const MB = 1024 * 1024;
const RUNS = 'data/hillclimb/runs.tsv';

const args = process.argv.slice(2);
const model = args[0] ?? 'small-upstream';
const precision = args[1] ?? 'f32';
const change = args[2] ?? 'baseline';
const ep = args[3] ?? 'webgpu';

function gitCommit() {
  return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
}

function appendRun(runId) {
  const line = [
    new Date().toISOString().slice(0, 10), runId, change, gitCommit(),
    'ort-web', model, 'L128K16', precision,
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
  if (!existsSync('bench/results/calibration-latest.json')) {
    throw new Error('missing calibration-latest.json; run bench/run-calibrate.mjs first');
  }
  const calibration = JSON.parse(readFileSync('bench/results/calibration-latest.json', 'utf8'));
  const metric = calibration.chosen;
  const metricFn = METRICS[metric];
  const runId = `ort-${model}-${precision}-${Date.now()}`;
  appendRun(runId);
  mkdirSync('bench/results', { recursive: true });

  let flags = calibration.flags ?? [];
  const browser = await chromium.launch({ headless: false, args: flags });
  const samples = [];
  let timer = null;
  let result;
  try {
    const page = await browser.newPage();
    // Let leftovers of the previous browser instance exit so the baseline is
    // not inflated by dying processes (pids are matched by binary path).
    await new Promise((r) => setTimeout(r, 1500));
    const tick = async () => {
      try {
        const pids = ownTreePids();
        samples.push({ t: Date.now(), pids: pids.length, bytes: metricFn(pids) });
      } catch { /* process tree mid-change */ }
    };
    timer = setInterval(tick, 100);
    await page.goto(
      `${BASE}/bench/ort.html?model=${model}&precision=${precision}&ep=${ep}`);
    const t0 = Date.now();
    while (Date.now() - t0 < 600000) {
      const done = await page.evaluate(() => window.khOrtResult?.done ?? false);
      if (done) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    await tick();
    clearInterval(timer);
    result = await page.evaluate(() => window.khOrtResult);
    if (result?.error) throw new Error('page: ' + result.error);
    const adapterInfo = result?.adapterInfo ?? null;
    if (!adapterInfo && flags.length === 0 && ep === 'webgpu') {
      throw new Error('no WebGPU adapter; retry with flags in calibration');
    }
    const browserVersion = browser.version();
    const baselineMb = median(samples.slice(0, 10).map((s) => s.bytes)) / MB;
    const peakMb = Math.max(...samples.map((s) => s.bytes)) / MB;
    const out = {
      date: new Date().toISOString().slice(0, 10),
      run_id: runId,
      engine: 'ort-web',
      model,
      precision,
      bucket: 'L128K16',
      browser: `chromium-${browserVersion}`,
      flags,
      adapterInfo,
      memMetric: metric,
      baselineMb,
      peakMemMb: peakMb - baselineMb,
      loadMs: result.loadMs,
      downloadBytes: result.downloadBytes,
      onnxBytes: result.onnxBytes,
      latency: result.latency,
      parity: result.parity,
      n: result.n,
      sampleCount: samples.length,
      sampleIntervalMs: median(
        samples.slice(1).map((s, i) => s.t - samples[i].t)).toFixed(0),
    };
    const file = `bench/results/${out.date}-ort-${model}-${precision}${ep === 'wasm' ? '-wasm' : ''}.json`;
    writeFileSync(file, JSON.stringify(out, null, 1));
    finishRun(runId, {
      argmax_agreement: out.parity.argmaxAgreement.toFixed(4),
      max_abs_logit_diff: out.parity.maxAbsLogitDiff.toExponential(2),
      max_abs_prob_diff: out.parity.maxAbsProbDiff.toExponential(2),
      median_ms: out.latency.medianMs.toFixed(2),
      p95_ms: out.latency.p95Ms.toFixed(2),
      model_only_median_ms: out.latency.medianMs.toFixed(2),
      load_ms: out.loadMs.toFixed(0),
      download_mb: (out.downloadBytes / MB).toFixed(1),
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
