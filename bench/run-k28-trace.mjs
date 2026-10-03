// K27 diagnosis runner for bench/k28-ort-trace.html (ORT shaders, dispatches, GPU time per
// pass) and bench/k28-kh-profile.html (kleinhirn per-dispatch profile). One visible Chromium,
// one runs.tsv line (change k27-diag, kept diagnostic), result JSON in bench/results/.
// Needs a Vite server with bench/k28/vite.k28-8.config.ts on <port>.
// Usage: node bench/run-k28-trace.mjs ort <slug> <L> <graph> <build> [port]
//        node bench/run-k28-trace.mjs kh <slug> <L> [port] [limits=minimum|default]

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { appendFileSync, writeFileSync } from 'node:fs';
import { loadavg } from 'node:os';

const [side, slug, L, a4, a5, a6] = process.argv.slice(2);
const ort = side === 'ort';
const port = (ort ? a6 : a4) ?? '5310';
const url = ort
  ? `/bench/k28-ort-trace.html?model=${slug}&L=${L}&graph=${a4}&build=${a5}`
  : `/bench/k28-kh-profile.html?model=${slug}&L=${L}&limits=${a5 ?? 'minimum'}`;
const global = ort ? 'khK28OrtTrace' : 'khK28KhProfile';
const commit = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
const tag = ort ? `${slug}-L${L}-${a4}-${a5}` : `${slug}-L${L}-${a5 ?? 'minimum'}`;
const runId = `k27-diag-${side}-${tag}-${Date.now()}`;
const file = `bench/results/k27-diag-${side}-${tag}.json`;
appendFileSync('data/hillclimb/runs.tsv', [
  new Date().toISOString().slice(0, 10), runId, 'k27-diag', commit, ort ? 'ort-web' : 'kleinhirn', slug, `L${L}`, 'f16',
  '', '', '', '', '', '', '', '', '', '', '', '', 'diagnostic', `${file}; load ${loadavg()[0].toFixed(2)}`,
].join('\t') + '\n');
// Without the flag Chromium quantizes timestamps to 65.5 µs (FINDINGS §17).
const browser = await chromium.launch({ headless: false, args: ['--enable-webgpu-developer-features'] });
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') console.log('[page]', m.text().slice(0, 200)); });
  await page.goto(`http://localhost:${port}${url}`);
  await page.waitForFunction((g) => window[g]?.done, global, { timeout: 20 * 60000 });
  const result = await page.evaluate((g) => window[g], global);
  writeFileSync(file, JSON.stringify({ run_id: runId, commit, browser: `chromium-${browser.version()}`,
    loadStart: loadavg()[0], ...result }, null, 1));
  console.log(file, result.stage, result.error ?? '');
} finally {
  await browser.close();
}
