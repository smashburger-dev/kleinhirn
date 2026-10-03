// K27 runner for bench/k27-mm.html (matmul variants on the encoder shapes). One visible
// Chromium with --enable-webgpu-developer-features (fine timestamps), one runs.tsv line
// (change k27-mm, kept diagnostic), result in bench/results/k27-mm-<tag>.json.
// Usage: node bench/run-k27-mm.mjs <tag> <variants> [shapes] [port]

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { appendFileSync, writeFileSync } from 'node:fs';
import { loadavg } from 'node:os';

const [tag, variants, shapes = '128x2304x768,128x768x768,128x3072x768,128x768x3072', port = '5310'] = process.argv.slice(2);
const commit = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
const runId = `k27-mm-${tag}-${Date.now()}`;
const file = `bench/results/k27-mm-${tag}.json`;
appendFileSync('data/hillclimb/runs.tsv', [
  new Date().toISOString().slice(0, 10), runId, 'k27-mm', commit, 'kleinhirn', 'matmul', shapes, 'f16',
  '', '', '', '', '', '', '', '', '', '', '', '', 'diagnostic', `${file}; ${variants}; load ${loadavg()[0].toFixed(2)}`,
].join('\t') + '\n');
const browser = await chromium.launch({ headless: false, args: ['--enable-webgpu-developer-features'] });
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') console.log('[page]', m.text().slice(0, 300)); });
  await page.goto(`http://localhost:${port}/bench/k27-mm.html?variants=${variants}&shapes=${shapes}`);
  await page.waitForFunction(() => window.khK27Mm?.done, null, { timeout: 20 * 60000 });
  const r = await page.evaluate(() => window.khK27Mm);
  writeFileSync(file, JSON.stringify({ run_id: runId, commit, variants, shapes, load: loadavg()[0], ...r }, null, 1));
  if (r.error) console.log('ERROR', r.error);
  const by = {};
  for (const row of r.rows) (by[row.shape] ??= []).push(row);
  for (const [shape, rows] of Object.entries(by)) {
    console.log(shape, rows.map((x) => `${x.variant} ${x.error ? 'ERR ' + x.error.slice(0, 80) : `${x.medianUs.toFixed(0)}us eq${(x.bitEqual * 100).toFixed(1)}% d${x.maxAbsDiff.toExponential(1)}`}`).join(' | '));
  }
} finally {
  await browser.close();
}
