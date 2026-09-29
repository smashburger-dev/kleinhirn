// Wide-schema parity runner: bench/wide.html in visible Chromium.
// Usage: node bench/run-wide.mjs <model> <precision>
import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const BASE = 'http://localhost:5199';
const RUNS = 'data/hillclimb/runs.tsv';
const model = process.argv[2] ?? 'base-upstream';
const precision = process.argv[3] ?? 'f32';
const limits = process.argv[4] ?? 'minimum';

function gitCommit() {
  return execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
}
function appendRun(runId) {
  const line = [
    new Date().toISOString().slice(0, 10), runId, 'wide-parity', gitCommit(),
    'kleinhirn', model, 'L1280K80', precision,
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
  const runId = `kh-wide-${model}-${precision}-${Date.now()}`;
  appendRun(runId);
  mkdirSync('bench/results', { recursive: true });
  const browser = await chromium.launch({ headless: false, args: [] });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.log('[page]', m.text());
    });
    await page.goto(
      `${BASE}/bench/wide.html?model=${model}&precision=${precision}&limits=${limits}`);
    const t0 = Date.now();
    while (Date.now() - t0 < 600000) {
      const done = await page.evaluate(() => window.khWideResult?.done ?? false);
      if (done) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    const result = await page.evaluate(() => window.khWideResult);
    if (result?.error) throw new Error('page: ' + result.error);
    const loadAvg = loadavg().map((v) => v.toFixed(2)).join(' ');
    const out = {
      date: new Date().toISOString().slice(0, 10),
      run_id: runId, engine: 'kleinhirn', model, precision,
      bucket: 'L1280K80', browser: `chromium-${browser.version()}`,
      loadAvg, ...result,
    };
    delete out.done;
    const file = `bench/results/${out.date}-wide-${model}-${precision}.json`;
    writeFileSync(file, JSON.stringify(out, null, 1));
    finishRun(runId, {
      argmax_agreement: (result.argmaxAgreement ?? NaN).toFixed(4),
      max_abs_logit_diff: (result.maxAbsLogitDiff ?? NaN).toExponential(2),
      max_abs_prob_diff: (result.maxAbsProbDiff ?? NaN).toExponential(2),
      gpu_mb: ((result.info?.gpuBytes ?? 0) / 1048576).toFixed(0),
      browser: out.browser,
      adapter: JSON.stringify(result.info?.adapter ?? {}).replace(/\t/g, ' '),
      kept: '',
      note: file,
    });
    console.log(JSON.stringify(result, null, 1));
  } finally {
    await browser.close();
  }
}

await main();
