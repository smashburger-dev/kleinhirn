// Debug runner for bench/julia-debug.html: prints the page result JSON.
// Usage: node bench/run-julia-debug.mjs [precision] [case]

import { chromium } from '@playwright/test';

const BASE = 'http://localhost:5199';
const precision = process.argv[2] ?? 'f32';
const caseIdx = process.argv[3] ?? '0';

const browser = await chromium.launch({ headless: false, args: [] });
try {
  const page = await browser.newPage();
  page.on('console', (m) => console.log('[page]', m.text()));
  const bucket = process.argv[4] ?? '';
  await page.goto(
    `${BASE}/bench/julia-debug.html?precision=${precision}&case=${caseIdx}`
    + `${bucket ? `&bucket=${bucket}` : ''}`);
  const t0 = Date.now();
  while (Date.now() - t0 < 300000) {
    const stage = await page.evaluate(
      () => window.khJuliaDebug?.stage ?? 'boot');
    if (stage === 'done' || stage === 'error') break;
    await new Promise((r) => setTimeout(r, 500));
  }
  const result = await page.evaluate(() => window.khJuliaDebug);
  console.log(JSON.stringify(result, null, 1));
} finally {
  await browser.close();
}
