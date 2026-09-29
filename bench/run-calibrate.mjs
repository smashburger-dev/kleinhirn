// Memory-metric calibration run (docs/ARCHITECTURE.md, Messprotokoll):
// opens bench/calibrate.html in visible Playwright Chromium, samples the
// process tree (browser PID + children) every 100 ms with all candidate
// metrics, and keeps the ones whose sum rises by 450-600 MB when the page
// allocates a 512 MB GPUBuffer. Writes bench/results/calibration-<date>.json
// and updates calibration-latest.json (the choice used by run-ort.mjs).

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { median, METRICS, sampleSeries } from './mem.mjs';

const BASE = 'http://localhost:5199';
const METRIC_NAMES = Object.keys(METRICS);
const MB = 1024 * 1024;

async function stage(page) {
  return page.evaluate(() => window.khCalibrate?.stage ?? 'missing');
}

async function waitStage(page, want, timeoutMs = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const s = await stage(page);
    if (s === want) return s;
    if (s === 'error') throw new Error('page error: ' + (await page.evaluate(() => window.khCalibrate?.error)));
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`timeout waiting for stage ${want}, at ${await stage(page)}`);
}

async function run(flags) {
  const browser = await chromium.launch({ headless: false, args: flags });
  try {
    const page = await browser.newPage();
    await page.goto(`${BASE}/bench/calibrate.html`);
    await waitStage(page, 'ready');

    const baseline = await sampleSeries(METRIC_NAMES, 10, 100);
    await page.evaluate(() => window.khCalibrate.alloc());
    await waitStage(page, 'allocated');
    const allocated = await sampleSeries(METRIC_NAMES, 30, 100);
    await page.evaluate(() => window.khCalibrate.release());
    await waitStage(page, 'released');
    const released = await sampleSeries(METRIC_NAMES, 10, 100);

    const adapterInfo = await page.evaluate(() => window.khCalibrate?.adapterInfo ?? null);
    return {
      browserVersion: browser.version(),
      adapterInfo,
      baseline, allocated, released,
    };
  } finally {
    await browser.close();
  }
}

async function main() {
  mkdirSync('bench/results', { recursive: true });
  let flags = [];
  let data = await run(flags);
  if (!data.adapterInfo) {
    flags = ['--enable-unsafe-webgpu'];
    data = await run(flags);
  }
  const deltas = {};
  const intervalOf = (rows) => median(rows.slice(1).map((r, i) => r.t - rows[i].t));
  const sampleIntervalMs = {
    baseline: intervalOf(data.baseline),
    allocated: intervalOf(data.allocated),
    released: intervalOf(data.released),
  };
  const valid = [];
  for (const name of METRIC_NAMES) {
    const base = median(data.baseline.map((r) => r[name]).filter((v) => v != null));
    const high = median(data.allocated.map((r) => r[name]).filter((v) => v != null));
    const after = median(data.released.map((r) => r[name]).filter((v) => v != null));
    deltas[name] = {
      baselineMb: base / MB,
      allocatedMb: high / MB,
      releasedMb: after / MB,
      deltaMb: (high - base) / MB,
    };
    if (high - base >= 450 * MB && high - base <= 600 * MB) valid.push(name);
  }
  const chosen = valid.includes('footprint') ? 'footprint' : (valid[0] ?? null);
  const date = new Date().toISOString().slice(0, 10);
  const out = {
    date,
    browser: `chromium-${data.browserVersion}`,
    flags,
    adapterInfo: data.adapterInfo,
    allocMb: 512,
    requiredDeltaMb: [450, 600],
    deltas,
    valid,
    chosen,
    sampleIntervalMs,
    sampleCounts: {
      baseline: data.baseline.length,
      allocated: data.allocated.length,
      released: data.released.length,
    },
    samples: data,
  };
  const file = `bench/results/calibration-${date}.json`;
  writeFileSync(file, JSON.stringify(out, null, 1));
  writeFileSync('bench/results/calibration-latest.json', JSON.stringify(out, null, 1));
  console.log(JSON.stringify({ file, deltas, valid, chosen }, null, 1));
  if (!chosen) {
    console.error('no metric passed the 450-600 MB check');
    process.exit(1);
  }
}

await main();
