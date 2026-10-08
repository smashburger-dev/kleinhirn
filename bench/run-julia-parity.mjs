// Julia 1 parity runner: visible Playwright Chromium on
// bench/julia-parity.html, writes bench/results/<date>-parity-julia-1-
// <precision>.json and a runs.tsv line.
// Usage: node bench/run-julia-parity.mjs <precision> [limits] [buckets] [backend]
// backend wasm (R2 stage 4): the WASM plan executor, f32 manifest, result file ...-wasm.json.

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';

const BASE = 'http://localhost:5199';
const RUNS = 'data/hillclimb/runs.tsv';

const precision = process.argv[2] ?? 'f32';
const limits = process.argv[3] ?? 'minimum';
const buckets = process.argv[4] ?? '';
const backend = process.argv[5] ?? '';

function gitCommit() {
  return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
}

function appendRun(runId) {
  const line = [
    new Date().toISOString().slice(0, 10), runId, 'parity-check', gitCommit(),
    'kleinhirn', 'julia-1', 'L512L1024K20', precision,
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
  const runId = `kh-parity-julia-1-${precision}${backend ? `-${backend}` : ''}-${limits}-${Date.now()}`;
  appendRun(runId);
  mkdirSync('bench/results', { recursive: true });
  const browser = await chromium.launch({ headless: false,
    args: (process.env.CHROMIUM_ARGS ?? '').split(' ').filter(Boolean) });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.log('[page]', m.text());
    });
    await page.goto(
      `${BASE}/bench/julia-parity.html?precision=${precision}`
      + `&limits=${limits}${buckets ? `&buckets=${buckets}` : ''}${backend ? `&backend=${backend}` : ''}`);
    const t0 = Date.now();
    while (Date.now() - t0 < 900000) {
      const done = await page.evaluate(() => window.khJuliaParityResult?.done ?? false);
      if (done) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    const result = await page.evaluate(() => window.khJuliaParityResult);
    if (result?.error) throw new Error('page: ' + result.error);
    const loadAvg = loadavg().map((v) => v.toFixed(2)).join(' ');
    const out = {
      date: new Date().toISOString().slice(0, 10),
      run_id: runId,
      engine: 'kleinhirn',
      model: 'julia-1',
      precision,
      browser: `chromium-${browser.version()}`,
      loadAvg, // official runs need the 1-min value <= 4
      ...result,
    };
    delete out.done;
    const file = `bench/results/${out.date}-parity-julia-1-${precision}${backend ? `-${backend}` : ''}.json`;
    writeFileSync(file, JSON.stringify(out, null, 1));
    const lg = result.logits ?? {};
    finishRun(runId, {
      argmax_agreement: (lg.argmaxAgreement ?? NaN).toFixed(4),
      max_abs_logit_diff: (lg.maxAbsLogitDiff ?? NaN).toExponential(2),
      max_abs_prob_diff: (lg.maxAbsProbDiff ?? NaN).toExponential(2),
      load_ms: (result.info?.loadMs ?? NaN).toFixed(0),
      download_mb: ((result.info?.downloadBytes ?? 0) / 1048576).toFixed(0),
      gpu_mb: ((result.info?.gpuBytes ?? 0) / 1048576).toFixed(0),
      browser: out.browser,
      adapter: JSON.stringify(result.adapterInfo ?? {}).replace(/\t/g, ' '),
      kept: '',
      note: file,
    });
    console.log(JSON.stringify({
      gates: result.gates,
      logits: result.logits,
      layers: result.layers,
    }, null, 1));
  } finally {
    await browser.close();
  }
}

await main();
