// Parity runner: visible Playwright Chromium on bench/parity.html, writes
// bench/results/<date>-parity-<model>-<precision>.json and a runs.tsv line.
// Usage: node bench/run-parity.mjs <model> <precision>

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';
import { dirname } from 'node:path';

const BASE = 'http://localhost:5199';
const RUNS = 'data/hillclimb/runs.tsv';

const model = process.argv[2] ?? 'small-upstream';
const precision = process.argv[3] ?? 'f32';
const limits = process.argv[4] ?? 'minimum';
const buckets = process.argv[5] ?? '';
const sets = process.argv[6] ?? '';
const backend = process.argv[7] ?? '';

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
  const bucketTag = sets === 'long'
    ? 'L512L1024K16'
    : 'L128L256K16';
  const line = [
    new Date().toISOString().slice(0, 10), runId, 'parity-check', gitCommit(),
    'kleinhirn', model, bucketTag, precision,
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
  const runId = `kh-parity-${model}-${precision}-${limits}-${Date.now()}`;
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
      `${BASE}/bench/parity.html?model=${model}&precision=${precision}`
      + `&limits=${limits}${buckets ? `&buckets=${buckets}` : ''}`
      + `${sets ? `&sets=${sets}` : ''}${backend ? `&backend=${backend}` : ''}`);
    const t0 = Date.now();
    while (Date.now() - t0 < 900000) {
      const done = await page.evaluate(() => window.khParityResult?.done ?? false);
      if (done) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    const result = await page.evaluate(() => window.khParityResult);
    if (result?.error) throw new Error('page: ' + result.error);
    const loadAvg = loadavg().map((v) => v.toFixed(2)).join(' ');
    const out = {
      date: new Date().toISOString().slice(0, 10),
      run_id: runId,
      engine: 'kleinhirn',
      model,
      precision,
      browser: `chromium-${browser.version()}`,
      loadAvg, // official runs need the 1-min value <= 4
      ...result,
    };
    delete out.done;
    const tag = backend ? `-${backend}` : '';
    const file = `bench/results/${out.date}-parity-${model}-${precision}${tag}.json`;
    writeFileSync(file, JSON.stringify(out, null, 1));
    // For sets=long there is no corpus block; record the worst values over
    // the L512/L1024 subsets so the tsv row carries the gate numbers.
    let corpusSet = result.corpus ?? {};
    if (result.long) {
      const ls = Object.values(result.long);
      corpusSet = {
        argmaxAgreement: Math.min(...ls.map((s) => s.argmaxAgreement)),
        maxAbsLogitDiff: Math.max(...ls.map((s) => s.maxAbsLogitDiff)),
        maxAbsProbDiff: Math.max(...ls.map((s) => s.maxAbsProbDiff)),
      };
    }
    finishRun(runId, {
      argmax_agreement: (corpusSet.argmaxAgreement ?? NaN).toFixed(4),
      max_abs_logit_diff: (corpusSet.maxAbsLogitDiff ?? NaN).toExponential(2),
      max_abs_prob_diff: (corpusSet.maxAbsProbDiff ?? NaN).toExponential(2),
      gpu_mb: ((result.info?.gpuBytes ?? 0) / 1048576).toFixed(0),
      browser: out.browser,
      adapter: JSON.stringify(result.adapterInfo ?? {}).replace(/\t/g, ' '),
      kept: '',
      note: file,
    });
    console.log(JSON.stringify({
      gates: result.gates,
      layers: result.layers,
      corpus: result.corpus,
      twoTasks: result.twoTasks,
    }, null, 1));
  } finally {
    await browser.close();
  }
}

await main();
