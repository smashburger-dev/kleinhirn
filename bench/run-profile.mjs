// K27 step 0: GPU time per dispatch. Opens bench/kleinhirn.html (small
// models) or bench/julia-bench.html (julia-1) with profile=dispatch, which
// times every dispatch in its own compute pass at the item's real seqLen and
// records the normal-path wall time of the same items. Chromium quantizes
// GPU timestamps by default (observed: 65.536 us); when the first run shows
// every value on a grid of its smallest nonzero value (>= 10 us), the run is
// repeated with
// --enable-webgpu-developer-features and the flag is recorded.
// Not an official latency run: it writes one runs.tsv line (kept diagnostic).
// Without --provisional it waits (up to 60 min) for a 1-minute load < 4 and
// aborts with the load and the top processes otherwise.
// Usage: node bench/run-profile.mjs <small-upstream|julia-1> <f16|f32> [--provisional]
// Output: bench/results/k27-profile-<model>-<precision>.json

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { loadavg } from 'node:os';

const BASE = 'http://localhost:5199';
const MAX_LOAD = 4;
const WAIT_MINUTES = 60;
const DEV_FLAG = '--enable-webgpu-developer-features';

const [model, precision] = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const provisional = process.argv.includes('--provisional');
if (!['small-upstream', 'julia-1'].includes(model) || !['f16', 'f32'].includes(precision)) {
  throw new Error('usage: run-profile.mjs <small-upstream|julia-1> <f16|f32> [--provisional]');
}
const julia = model === 'julia-1';
const file = `bench/results/k27-profile-${model}-${precision}.json`;
const commit = execFileSync(
  'git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();

appendFileSync('data/hillclimb/runs.tsv', [
  new Date().toISOString().slice(0, 10), `k27-profile-${model}-${precision}-${Date.now()}`,
  'k27-profile', commit, julia ? 'kleinhirn-julia' : 'kleinhirn', model,
  julia ? 'julia' : 'L128K16', precision,
  '', '', '', '', '', '', '', '', '', '', '', '', 'diagnostic', file,
].join('\t') + '\n');

async function buildIdCheck() {
  if (julia) return null;
  const disk = readFileSync('dist/kleinhirn.js', 'utf8');
  const id = disk.match(/buildId\s*:\s*["']([a-z0-9]+)["']/)?.[1];
  if (!id) throw new Error('no buildId in dist/kleinhirn.js; run npm run build');
  const served = await (await fetch(`${BASE}/dist/kleinhirn.js`)).text();
  if (!served.includes(id)) {
    throw new Error(`server ${BASE} serves a stale bundle (no buildId ${id})`);
  }
  return id;
}

async function waitForLoad() {
  const t0 = Date.now();
  for (;;) {
    const l = loadavg()[0];
    if (l < MAX_LOAD) return l;
    if (Date.now() - t0 > WAIT_MINUTES * 60000) {
      const top = execFileSync(
        'ps', ['-Ao', 'pid,pcpu,etime,comm', '-r'], { encoding: 'utf8' })
        .split('\n').slice(0, 10).join('\n');
      throw new Error(`load stayed >= ${MAX_LOAD} for ${WAIT_MINUTES} min (now ${l})\n${top}`);
    }
    process.stdout.write(`waiting for load < ${MAX_LOAD}: ${l}\n`);
    await new Promise((r) => setTimeout(r, 30000));
  }
}

async function runOnce(args) {
  const browser = await chromium.launch({ headless: false, args });
  try {
    const page = await browser.newPage();
    const url = julia
      ? `${BASE}/bench/julia-bench.html?precision=${precision}&limits=minimum&profile=dispatch`
      : `${BASE}/bench/kleinhirn.html?model=${model}&precision=${precision}`
        + '&limits=minimum&buckets=128&profile=dispatch';
    await page.goto(url);
    await page.waitForFunction(
      () => (window.khResult ?? window.khJuliaBench)?.done, null, { timeout: 1800000 });
    const page_ = await page.evaluate(() => window.khResult ?? window.khJuliaBench);
    if (page_.error) throw new Error(`page: ${page_.error}`);
    return { page: page_, browser: `chromium-${browser.version()}` };
  } finally {
    await browser.close();
  }
}

const loadAvgStart = provisional ? loadavg()[0] : await waitForLoad();
const buildId = await buildIdCheck();
let run = await runOnce([]);
let flags = [];
let withoutFlag = null;
const q = run.page.dispatchProfile?.quantization;
if (q?.quantized) {
  withoutFlag = q;
  flags = [DEV_FLAG];
  run = await runOnce(flags);
}
const loadAvgEnd = loadavg()[0];
mkdirSync('bench/results', { recursive: true });
writeFileSync(file, JSON.stringify({
  date: new Date().toISOString().slice(0, 10), commit, buildId, model, precision,
  browser: run.browser, flags, quantizationWithoutFlag: withoutFlag,
  official: !provisional && loadAvgStart < MAX_LOAD,
  loadAvgStart, loadAvgEnd, page: run.page,
}, null, 1));
const dp = run.page.dispatchProfile;
console.log(file, dp ? {
  gpuSumMedianMs: dp.gpuSumMedianMs, wallMedianMs: dp.wallMedianMs,
  parity: dp.paritySample, quantization: dp.quantization, flags,
} : 'no timestamp-query');
