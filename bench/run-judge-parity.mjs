// Judge parity / countdown gate runner (K17): visible Playwright
// Chromium on bench/judge-parity.html, writes
// bench/results/<date>-judge-<mode>-<precision>.json and a runs.tsv line.
// Usage: node bench/run-judge-parity.mjs [mode] [precision] [limits] [port]

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';
import { dirname } from 'node:path';

const mode = process.argv[2] ?? 'parity';
const precision = process.argv[3] ?? 'f32';
const limits = process.argv[4] ?? 'minimum';
const BASE = `http://localhost:${process.argv[5] ?? '5210'}`;
const RUNS = 'data/hillclimb/runs.tsv';

function gitCommit() {
  return execFileSync(
    'git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
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
    new Date().toISOString().slice(0, 10), runId, `judge-${mode}`,
    gitCommit(), 'kleinhirn', 'countdown-judge', '', precision,
    '', '', '', '', '', '', '', '', '', '', '', '', '', 'pending',
  ].join('\t');
  appendFileSync(RUNS, line + '\n');
}

function finishRun(runId, fields) {
  const lines = readFileSync(RUNS, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  const cells = lines[idx].split('\t');
  const cols = [
    'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket',
    'precision', 'argmax_agreement', 'max_abs_logit_diff',
    'max_abs_prob_diff', 'median_ms', 'p95_ms', 'model_only_median_ms',
    'load_ms', 'download_mb', 'gpu_mb', 'peak_mem_mb', 'browser',
    'adapter', 'kept', 'note',
  ];
  for (const [k, v] of Object.entries(fields)) cells[cols.indexOf(k)] = String(v);
  lines[idx] = cells.join('\t');
  writeFileSync(RUNS, lines.join('\n'));
}

async function main() {
  const runId = `kh-judge-${mode}-${precision}-${Date.now()}`;
  appendRun(runId);
  mkdirSync('bench/results', { recursive: true });
  const browser = await chromium.launch({ headless: false, args: [] });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.log('[page]', m.text());
    });
    await page.goto(
      `${BASE}/bench/judge-parity.html?mode=${mode}&precision=${precision}`
      + `&limits=${limits}`);
    const t0 = Date.now();
    while (Date.now() - t0 < 1800000) {
      const done = await page.evaluate(
        () => window.khJudgeResult?.done ?? false);
      if (done) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    const result = await page.evaluate(() => window.khJudgeResult);
    if (result?.error) throw new Error('page: ' + result.error);
    const loadAvg = loadavg().map((v) => v.toFixed(2)).join(' ');
    const out = {
      date: new Date().toISOString().slice(0, 10),
      run_id: runId,
      engine: 'kleinhirn',
      model: 'countdown-judge',
      mode,
      precision,
      browser: `chromium-${browser.version()}`,
      loadAvg,
      ...result,
    };
    delete out.done;
    const file = `bench/results/${out.date}-judge-${mode}-${precision}.json`;
    writeFileSync(file, JSON.stringify(out, null, 1));
    const fields = {
      browser: out.browser,
      gpu_mb: ((result.info?.gpuBytes ?? 0) / 1048576).toFixed(1),
      kept: Object.values(result.gates ?? {}).every(Boolean)
        ? 'parity' : 'parity-fail',
      note: file,
    };
    if (mode === 'parity') {
      fields.max_abs_logit_diff = result.parity?.maxAbsLogitDiff
        ?.toExponential(2) ?? '';
      fields.max_abs_prob_diff = result.parity?.maxAbsProbDiff
        ?.toExponential(2) ?? '';
      fields.argmax_agreement = result.parity?.decisionAgreement
        ?.toFixed(4) ?? '';
    } else {
      fields.argmax_agreement = `${result.tasks?.solved}/${result.tasks?.total}`;
      fields.median_ms = result.tasks?.msPerTask?.median?.toFixed(1) ?? '';
    }
    finishRun(runId, fields);
    console.log(JSON.stringify(result.gates, null, 1));
  } finally {
    await browser.close();
  }
}

await main();
