// K9 typed-decisions eval runner for kleinhirn-Julia (browser, WebGPU).
// Usage: node bench/run-julia-k9.mjs [precision] [limit]
// Requires the dev server on :5199. Writes
// bench/results/k9-typed-kleinhirn-julia-<precision>.json and a runs.tsv line.

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';

const BASE = 'http://localhost:5199';
const RUNS = 'data/hillclimb/runs.tsv';
const precision = process.argv[2] ?? 'f16';
const limit = process.argv[3] ?? '';

function gitCommit() {
  return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
}

function appendRun(runId) {
  const line = [
    new Date().toISOString().slice(0, 10), runId, 'k9-typed-eval', gitCommit(),
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
  const runId = `kh-k9-typed-${precision}-${Date.now()}`;
  appendRun(runId);
  mkdirSync('bench/results', { recursive: true });
  const browser = await chromium.launch({ headless: false, args: [] });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.log('[page]', m.text());
    });
    await page.goto(
      `${BASE}/bench/julia-k9.html?precision=${precision}`
      + `${limit ? `&limit=${limit}` : ''}`);
    const t0 = Date.now();
    while (Date.now() - t0 < 3600000) {
      const done = await page.evaluate(() => window.khK9Result?.done ?? false);
      if (done) break;
      await new Promise((r) => setTimeout(r, 2000));
    }
    const result = await page.evaluate(() => window.khK9Result);
    if (result?.error) throw new Error('page: ' + result.error);
    const loadAvg = loadavg().map((v) => v.toFixed(2)).join(' ');
    const predictions = result.predictions ?? [];
    const medianMs = predictions.length
      ? predictions.map((p) => p.latencyMs).sort((a, b) => a - b)[
        Math.floor(predictions.length / 2)]
      : NaN;
    const out = {
      task: 'typed',
      route: `kleinhirn-julia-webgpu-${precision}`,
      model: 'julia-1',
      date: new Date().toISOString().slice(0, 10),
      run_id: runId,
      browser: `chromium-${browser.version()}`,
      loadAvg,
      ...result,
    };
    delete out.done;
    const file = `bench/results/k9-typed-kleinhirn-julia-${precision}.json`;
    writeFileSync(file, JSON.stringify(out, null, 1));
    finishRun(runId, {
      argmax_agreement: '',
      median_ms: medianMs.toFixed(2),
      load_ms: (result.info?.loadMs ?? NaN).toFixed(0),
      download_mb: ((result.info?.downloadBytes ?? 0) / 1048576).toFixed(0),
      gpu_mb: ((result.info?.gpuBytes ?? 0) / 1048576).toFixed(0),
      browser: out.browser,
      adapter: JSON.stringify(result.info?.adapter ?? {}).replace(/\t/g, ' '),
      kept: '',
      note: file,
    });
    console.log(JSON.stringify({
      total: out.total, correct: out.correct,
      accuracy: out.accuracy, abstained: out.abstained,
      byType: out.byType, medianMs,
    }, null, 1));
  } finally {
    await browser.close();
  }
}

await main();
