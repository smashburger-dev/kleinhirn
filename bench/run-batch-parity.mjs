// Batch parity runner (K16): visible Playwright Chromium on
// bench/batch-parity.html, writes bench/results/<date>-batch-parity-
// <model>-<precision>.json and a runs.tsv line.
// Usage: node bench/run-batch-parity.mjs <model> <precision> [limits] [sizes]

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';
import { dirname } from 'node:path';

const BASE = `http://localhost:${process.argv[6] ?? '5199'}`;
const RUNS = 'data/hillclimb/runs.tsv';

const model = process.argv[2] ?? 'small-upstream';
const precision = process.argv[3] ?? 'f32';
const limits = process.argv[4] ?? 'minimum';
const sizes = process.argv[5] ?? '1,4,8,16';

function gitCommit() {
  return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
}

function appendRun(runId) {
  if (!existsSync(RUNS)) {
    mkdirSync(dirname(RUNS), { recursive: true });
    writeFileSync(RUNS, [
      'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket',
      'precision', 'argmax_agreement', 'max_abs_logit_diff',
      'max_abs_prob_diff', 'median_ms', 'p95_ms', 'model_only_median_ms',
      'load_ms', 'download_mb', 'gpu_mb', 'peak_mem_mb', 'browser',
      'adapter', 'kept', 'note',
    ].join('\t') + '\n');
  }
  const line = [
    new Date().toISOString().slice(0, 10), runId, 'batch-parity', gitCommit(),
    'kleinhirn', model, `B${sizes.replaceAll(',', '-')}`, precision,
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
  const runId = `kh-batch-parity-${model}-${precision}-${Date.now()}`;
  appendRun(runId);
  mkdirSync('bench/results', { recursive: true });
  const browser = await chromium.launch({ headless: false, args: [] });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.log('[page]', m.text());
    });
    await page.goto(
      `${BASE}/bench/batch-parity.html?model=${model}&precision=${precision}`
      + `&limits=${limits}&sizes=${sizes}`);
    const t0 = Date.now();
    while (Date.now() - t0 < 900000) {
      const done = await page.evaluate(() => window.khBatchResult?.done ?? false);
      if (done) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    const result = await page.evaluate(() => window.khBatchResult);
    if (result?.error) throw new Error('page: ' + result.error);
    const loadAvg = loadavg().map((v) => v.toFixed(2)).join(' ');
    const out = {
      date: new Date().toISOString().slice(0, 10),
      run_id: runId,
      engine: 'kleinhirn',
      model,
      precision,
      browser: `chromium-${browser.version()}`,
      loadAvg,
      ...result,
    };
    delete out.done;
    const file = `bench/results/${out.date}-batch-parity-${model}-${precision}.json`;
    writeFileSync(file, JSON.stringify(out, null, 1));
    const allSets = {
      ...(result.sizes ?? {}),
      ...(result.mixed ?? {}),
    };
    const worstSingle = Math.max(...Object.values(allSets).map(
      (s) => s.vsSingle?.maxAbsLogitDiff ?? 0));
    const minArgmax = Math.min(...Object.values(allSets).flatMap(
      (s) => [s.vsSingle?.argmaxAgreement ?? 1, s.vsGolden?.argmaxAgreement ?? 1]));
    finishRun(runId, {
      argmax_agreement: minArgmax.toFixed(4),
      max_abs_logit_diff: worstSingle.toExponential(2),
      gpu_mb: ((result.info?.gpuBytes ?? 0) / 1048576).toFixed(0),
      browser: out.browser,
      adapter: JSON.stringify(result.adapterInfo ?? {}).replace(/\t/g, ' '),
      kept: '',
      note: file,
    });
    console.log(JSON.stringify(result.gates, null, 1));
  } finally {
    await browser.close();
  }
}

await main();
