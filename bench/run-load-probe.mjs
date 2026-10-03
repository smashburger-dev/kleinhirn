// K28 provisional load-path probe runner. Starts bench/k28/vite.config.ts,
// then for each model x rep runs the variants A, Bcpu, Bgpu in a FRESH visible
// Playwright Chromium (interleaved per rep), sampling the OS memory of the
// runner's process tree (minus the vite server subtree) every 100 ms with the
// calibrated metric. All numbers are provisional ("vorläufig").
// Usage: node bench/run-load-probe.mjs [reps=3]

import { chromium } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { loadavg } from 'node:os';
import { gzipSync } from 'node:zlib';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { median, METRICS, ownTreePids } from './mem.mjs';

const PORT = 5288;
const BASE = `http://localhost:${PORT}`;
const MB = 1024 * 1024;
const RUNS = Number(process.argv[2] ?? 3);
const MODELS = ['small-upstream', 'julia-1'];
const VARIANTS = ['A', 'Bcpu', 'Bgpu'];
const BUSY = 'run-official|run-kleinhirn|run-ort|run-profile|run-parity|run-julia|run-batch';
const WAIT_BUDGET_MS = 30 * 60 * 1000;
const RESULTS = 'bench/results';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let waitedMs = 0;
async function waitClear() {
  for (;;) {
    let busy = '';
    try {
      busy = execFileSync('pgrep', ['-f', BUSY], { encoding: 'utf8' }).trim();
    } catch { /* no match */ }
    const load = loadavg()[0];
    if (!busy && load <= 6) return;
    if (waitedMs >= WAIT_BUDGET_MS) {
      throw new Error(`gave up after ${(waitedMs / 60000).toFixed(1)} min waiting `
        + `(busy pids: ${busy.replace(/\n/g, ',') || '-'}, load ${load.toFixed(2)})`);
    }
    console.log(`waiting: busy=[${busy.replace(/\n/g, ',')}] load1=${load.toFixed(2)}`);
    await sleep(30000);
    waitedMs += 30000;
  }
}

async function startVite() {
  const child = spawn('npx', ['vite', '--config', 'bench/k28/vite.config.ts'], {
    stdio: ['ignore', 'pipe', 'pipe'], detached: true,
  });
  let log = '';
  child.stdout.on('data', (d) => { log += d; });
  child.stderr.on('data', (d) => { log += d; });
  for (let i = 0; i < 100; i += 1) {
    try {
      const r = await fetch(`${BASE}/bench/k28/load-probe.html`);
      if (r.ok) return child;
    } catch { /* not up yet */ }
    await sleep(300);
  }
  throw new Error(`vite did not start:\n${log}`);
}

async function codeSizes() {
  const { build } = await import('vite');
  const gz = async (entry) => {
    const out = await build({
      configFile: false, logLevel: 'silent',
      build: {
        write: false, minify: 'oxc', target: 'es2022', // vite 8 minifier; esbuild is not installed
        lib: { entry, formats: ['es'], fileName: () => 'out.js' },
      },
    });
    const chunk = (Array.isArray(out) ? out[0] : out).output[0];
    return { raw: Buffer.byteLength(chunk.code), gzip: gzipSync(chunk.code, { level: 9 }).byteLength };
  };
  return {
    stLoaderGzip: (await gz('bench/k28/st-loader.ts')).gzip,
    weightsTsGzip: (await gz('src/weights.ts')).gzip,
    gpuPrepGzip: (await gz('bench/k28/gpu-prep.ts')).gzip,
    detail: {
      stLoader: await gz('bench/k28/st-loader.ts'),
      weightsTs: await gz('src/weights.ts'),
      gpuPrep: await gz('bench/k28/gpu-prep.ts'),
    },
  };
}

async function oneRun(model, variant, rep, viteTreeRoot, calibration) {
  await waitClear();
  const metricFn = METRICS[calibration.chosen];
  const loadAvgStart = loadavg()[0];
  const browser = await chromium.launch({
    headless: false, args: [...(calibration.flags ?? []),
      ...(process.env.CHROMIUM_ARGS ?? '').split(' ').filter(Boolean)],
  });
  const samples = [];
  let timer = null;
  try {
    const page = await browser.newPage();
    await sleep(1500);
    const tick = () => {
      try {
        const vite = new Set(ownTreePids(viteTreeRoot));
        const pids = ownTreePids().filter((p) => !vite.has(p));
        samples.push({ t: Date.now(), pids: pids.length, bytes: metricFn(pids) });
      } catch { /* process tree mid-change */ }
    };
    timer = setInterval(tick, 100);
    await sleep(1200); // pre-navigation baseline samples
    const nBase = samples.length;
    await page.goto(`${BASE}/bench/k28/load-probe.html?model=${model}&variant=${variant}`);
    const t0 = Date.now();
    for (;;) {
      const s = await page.evaluate(() => ({ p: !!window.__phase1, e: window.__error ?? null }));
      if (s.e) throw new Error(`page error: ${s.e}`);
      if (s.p) break;
      if (Date.now() - t0 > 600000) throw new Error('phase 1 timeout');
      await sleep(100);
    }
    await sleep(1000);
    clearInterval(timer);
    timer = null;
    tick();
    const baselineMb = median(samples.slice(0, nBase).map((s) => s.bytes)) / MB;
    const peakMb = Math.max(...samples.map((s) => s.bytes)) / MB;
    await page.evaluate(() => { window.__proceed = true; });
    const t1 = Date.now();
    let result = null;
    for (;;) {
      const s = await page.evaluate(() => ({ r: window.__result ?? null, e: window.__error ?? null }));
      if (s.e) throw new Error(`page error: ${s.e}`);
      if (s.r) { result = s.r; break; }
      if (Date.now() - t1 > 600000) throw new Error('checks timeout');
      await sleep(200);
    }
    const out = {
      date: new Date().toISOString(), model, variant, rep, provisional: true,
      loadAvg1Start: Number(loadAvgStart.toFixed(2)),
      loadAvg1End: Number(loadavg()[0].toFixed(2)),
      browser: `chromium-${browser.version()}`, flags: calibration.flags ?? [],
      memMetric: calibration.chosen, baselineMb, peakMemMB: peakMb - baselineMb,
      sampleCount: samples.length,
      sampleIntervalMs: Number(median(samples.slice(1).map((s, i) => s.t - samples[i].t)).toFixed(0)),
      ...result,
    };
    writeFileSync(`${RESULTS}/k28-load-probe-${model}-${variant}-${rep}.json`, JSON.stringify(out, null, 1));
    return out;
  } finally {
    if (timer) clearInterval(timer);
    await browser.close();
  }
}

const range = (xs) => {
  const v = xs.filter((x) => typeof x === 'number');
  return v.length ? { median: median(v), min: Math.min(...v), max: Math.max(...v) } : null;
};

async function main() {
  mkdirSync(RESULTS, { recursive: true });
  const calibration = JSON.parse(readFileSync(`${RESULTS}/calibration-latest.json`, 'utf8'));
  const vite = await startVite();
  const results = [];
  try {
    for (const model of MODELS) {
      for (let rep = 1; rep <= RUNS; rep += 1) {
        for (const variant of VARIANTS) {
          console.log(`run ${model} ${variant} rep ${rep}`);
          const r = await oneRun(model, variant, rep, vite.pid, calibration);
          results.push(r);
          console.log(`  total ${r.timing.totalMs.toFixed(0)} ms, peak ${r.peakMemMB.toFixed(0)} MB, load ${r.loadAvg1Start}`);
        }
      }
    }
  } finally {
    try { process.kill(-vite.pid, 'SIGTERM'); } catch { /* gone */ }
  }
  const summary = { provisional: true, date: new Date().toISOString(), reps: RUNS, waitedMs, cells: {} };
  for (const model of MODELS) {
    for (const variant of VARIANTS) {
      const rs = results.filter((r) => r.model === model && r.variant === variant);
      const keys = Object.keys(rs[0].timing);
      const cell = { runs: rs.length, timing: {}, peakMemMB: range(rs.map((r) => r.peakMemMB)),
        loadAvg1Start: rs.map((r) => r.loadAvg1Start) };
      for (const k of keys) cell.timing[k] = range(rs.map((r) => r.timing[k]));
      if (variant !== 'A') {
        cell.checks = rs.map((r) => ({
          rep: r.rep, tensorCount: r.checks.tensorCount, namesEqual: r.checks.namesEqual,
          shapeMismatches: r.checks.shapeMismatches.length, sha256Match: r.checks.sha256.match,
          gpuBytesEqual: r.checks.gpuBytes.equal,
          bitIdentical: `${r.checks.bitIdenticalTensors}/${r.checks.totalTensors}`,
          unexpectedDiffering: r.checks.unexpectedDiffering,
          precomputeAggregate: r.checks.precomputeAggregate,
        }));
      }
      summary.cells[`${model}/${variant}`] = cell;
    }
  }
  summary.codeSize = await codeSizes();
  writeFileSync(`${RESULTS}/k28-load-probe-summary.json`, JSON.stringify(summary, null, 1));
  console.log('summary written');
}

await main();
