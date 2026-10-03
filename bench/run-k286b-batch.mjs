// K28.6b G4 runner: visible Playwright Chromium on bench/k286b-batch.html, writes
// bench/results/k286b-batch-<slug>-<precision>-<commit>[-<label>].json and one runs.tsv line
// (change k28.6b).
// Usage: node bench/run-k286b-batch.mjs julia-1|<slug> <f32|f16> [port] [sizes] [--label=x] [--allow-dirty]

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const RUNS = 'data/hillclimb/runs.tsv';
const OUT = 'bench/results';
const git = (...a) => execFileSync('git', a, { encoding: 'utf8' }).trim();
const args = process.argv.slice(2);
const flags = args.filter((a) => a.startsWith('--'));
const [slug, precision, port = '5199', sizes = '8,16'] = args.filter((a) => !a.startsWith('--'));
const label = flags.find((f) => f.startsWith('--label='))?.slice(8) ?? '';
if (!slug || !['f32', 'f16'].includes(precision)) {
  console.error('usage: run-k286b-batch.mjs julia-1|<slug> <f32|f16> [port] [sizes] [--label=x]');
  process.exit(2);
}
const commit = git('rev-parse', '--short', 'HEAD');
const dirty = git('status', '--porcelain', '--', 'src', 'bench/k286b-batch.ts', 'bench/k286b-batch.html');
if (dirty && !flags.includes('--allow-dirty')) {
  console.error(`engine or page files are uncommitted, commit first:\n${dirty}`);
  process.exit(2);
}
const COLS = [
  'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket', 'precision',
  'argmax_agreement', 'max_abs_logit_diff', 'max_abs_prob_diff', 'median_ms',
  'p95_ms', 'model_only_median_ms', 'load_ms', 'download_mb', 'gpu_mb',
  'peak_mem_mb', 'browser', 'adapter', 'kept', 'note',
];
const kind = slug === 'julia-1' ? 'julia' : 'encoder';
const runId = `k286b-batch-${slug}-${precision}-${commit}${label ? `-${label}` : ''}-${Date.now()}`;
const cells = COLS.map(() => '');
Object.assign(cells, {
  0: new Date().toISOString().slice(0, 10), 1: runId, 2: 'k28.6b', 3: commit, 4: 'kleinhirn', 5: slug,
  6: `B${sizes.replaceAll(',', '-')}`, 7: precision, 21: 'pending',
});
appendFileSync(RUNS, cells.join('\t') + '\n');

function finishRun(fields) {
  const lines = readFileSync(RUNS, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  const c = lines[idx].split('\t');
  for (const [k, v] of Object.entries(fields)) c[COLS.indexOf(k)] = String(v);
  lines[idx] = c.join('\t');
  writeFileSync(RUNS, lines.join('\n'));
}

mkdirSync(OUT, { recursive: true });
const loadStart = loadavg()[0].toFixed(2);
const browser = await chromium.launch({ headless: false, args: [] });
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') console.log('[page]', m.text()); });
  await page.goto(`http://localhost:${port}/bench/k286b-batch.html?kind=${kind}&model=${slug}`
    + `&precision=${precision}&sizes=${sizes}`);
  const t0 = Date.now();
  while (Date.now() - t0 < 900000) {
    if (await page.evaluate(() => window.khK286bResult?.done ?? false)) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  const result = await page.evaluate(() => window.khK286bResult);
  if (!result?.done) throw new Error('timeout');
  const out = {
    date: new Date().toISOString().slice(0, 10), run_id: runId, commit,
    browser: `chromium-${browser.version()}`, loadAvgStart: loadStart,
    loadAvgEnd: loadavg().map((v) => v.toFixed(2)).join(' '), ...result,
  };
  delete out.done;
  const file = `${OUT}/k286b-batch-${slug}-${precision}-${commit}${label ? `-${label}` : ''}.json`;
  writeFileSync(file, JSON.stringify(out, null, 1));
  const sz = Object.values(result.sizes ?? {});
  const diff = sz.reduce((a, s) => a + s.differingRows, 0);
  const errs = sz.reduce((a, s) => a + s.gpuErrors.length, 0);
  finishRun({
    gpu_mb: ((result.info?.gpuBytes ?? 0) / 1048576).toFixed(1), browser: out.browser,
    adapter: JSON.stringify(result.adapterInfo ?? {}).replace(/\t/g, ' '),
    kept: result.error ? 'error' : '',
    note: `${file}; differing rows ${diff}; gpu errors ${errs}; load ${loadStart} -> ${loadavg()[0].toFixed(2)}${result.error ? `; ${result.error}` : ''}`,
  });
  console.log(file);
  console.log(JSON.stringify({ error: result.error, lengths: result.lengths, sizes: result.sizes }, null, 1));
} finally {
  await browser.close();
}
