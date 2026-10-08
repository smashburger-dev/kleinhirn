// R8 diagnosis runner (docs/R8_WORKORDER.md Phase 1): the matmul micro-bench (bench/r8-gemm.html)
// and the op profile (bench/r8-prof.html) per browser, on the bench server with isolation. One
// runs.tsv line per run (change r8-diag, search-run rule), result bench/results/r8-diag-*.json.
// Usage: node bench/run-r8-diag.mjs gemm [--browsers chromium,webkit,safari,firefox-reg]
//        node bench/run-r8-diag.mjs prof [--browsers ...] [--models minilm,roberta-base] [--lens 128,512] [--reps 20]
//        node bench/run-r8-diag.mjs batch [--browsers chromium] [--models ...] [--threads 8] (R8 hc9, change r8-hc9,
//        page bench/r8-batch.html on the built dist/, result bench/results/r8-hc9-batch-<browser>-<model>.json)

import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  MODELS, PauseError, RESULTS, ROOT, appendRun, checkEnvironment, finishRun, openBrowser, sleep, startVite, stopVite,
} from './kbench/runner-lib.mjs';

const args = process.argv.slice(2);
const mode = args[0];
const val = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const browsers = val('--browsers', 'chromium,webkit,safari,firefox-reg').split(',');
const shorts = Object.fromEntries(MODELS.map((m) => [m.short.toLowerCase(), m.slug]));
const models = val('--models', 'minilm,roberta-base').split(',').map((m) => shorts[m.toLowerCase()] ?? m);
const lens = val('--lens', '128,512');
const reps = val('--reps', '20');
const commit = execFileSync('git', ['-C', ROOT, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
const ctx = { ownPids: new Set([process.pid, process.ppid]), vite: null, base: null };

async function page(browser, url, global, runId) {
  const env = await checkEnvironment(ctx);
  const drv = await openBrowser(browser, global);
  try {
    await drv.goto(url);
    const t0 = Date.now();
    let st;
    while (Date.now() - t0 < 40 * 60000) {
      st = await drv.state();
      if (st.done) break;
      await sleep(browser === 'safari' ? 2000 : 1000);
    }
    if (!st?.done) throw new Error(`page timeout (${st?.stage})`);
    const r = await drv.result();
    if (st.error || r?.error) throw new Error(st.error ?? r.error);
    return { browser: drv.label, loadStart: env.loadStart, ...r };
  } catch (e) {
    finishRun(runId, { kept: 'error', note: String(e.message ?? e).slice(0, 300) });
    throw e;
  } finally {
    await drv.close();
  }
}

const f1 = (v) => (typeof v === 'number' ? v.toFixed(1) : '-');

async function gemm(browser) {
  const runId = `r8-diag-gemm-${browser}-${Date.now()}`;
  appendRun(runId, { change: 'r8-diag', commit, engine: 'gemm-probe', model: '-', bucket: 'M128', precision: 'wasm-f32', browser, note: 'R8 phase 1: matmul micro-bench v0..v6, plain and relaxed, loops' });
  const r = await page(browser, `${ctx.base}/bench/r8-gemm.html`, 'khR8Gemm', runId);
  const file = join(RESULTS, `r8-diag-gemm-${browser}.json`);
  writeFileSync(file, `${JSON.stringify({ date: new Date().toISOString(), commit, ...r }, null, 1)}\n`);
  const geo = (res, v) => Math.exp(res.shapes.reduce((s, x) => s + Math.log(x.gmacs[v]), 0) / res.shapes.length);
  const line = (res) => Object.keys(res.shapes[0].gmacs).map((v) => `${v} ${f1(geo(res, v))}`).join(' ');
  const errs = [r.plain, r.relaxed].filter(Boolean).flatMap((res) => res.shapes.flatMap((x) => Object.values(x.maxErr)));
  const maxErr = Math.max(...errs);
  console.log(`${r.browser}: loops scalar ${f1(r.loops.wasmScalarMs)} simd ${f1(r.loops.wasmSimdMs)} ms; peak ${f1(r.plain.peakGmacs)} GMAC/s`);
  console.log(`  plain   ${line(r.plain)}`);
  if (r.relaxed) console.log(`  relaxed ${line(r.relaxed)}`);
  console.log(`  max error ${maxErr.toExponential(1)}`);
  finishRun(runId, { kept: maxErr < 1e-3 ? 'diag' : 'error', browser: r.browser, note: `bench/results/r8-diag-gemm-${browser}.json; load ${r.loadStart}; v0 ${f1(geo(r.plain, 'v0'))} v3 ${f1(geo(r.plain, 'v3'))}${r.relaxed ? ` v3r ${f1(geo(r.relaxed, 'v3'))}` : ''} GMAC/s` });
}

async function prof(browser, slug) {
  const short = MODELS.find((m) => m.slug === slug).short;
  const runId = `r8-diag-prof-${browser}-${short}-${Date.now()}`;
  appendRun(runId, { change: 'r8-diag', commit, engine: 'kleinhirn', model: slug, bucket: `L${lens.replace(',', '+L')}`, precision: 'wasm-f32', browser, note: `R8 phase 1: op profile, reps ${reps}` });
  const r = await page(browser, `${ctx.base}/bench/r8-prof.html?model=${slug}&lens=${lens}&reps=${reps}`, 'khR8Prof', runId);
  writeFileSync(join(RESULTS, `r8-diag-prof-${browser}-${short}.json`), `${JSON.stringify({ date: new Date().toISOString(), commit, ...r }, null, 1)}\n`);
  for (const x of r.results) {
    const parts = Object.entries(x.byKernel).sort((a, b) => b[1].ms - a[1].ms)
      .map(([k, v]) => `${k} ${f1(v.ms)}${v.macs ? ` (${f1(v.macs / v.ms / 1e6)} GMAC/s)` : ''}`).join(', ');
    console.log(`${r.browser} ${short} L${x.len}: forward ${f1(x.profile.totalMs)} ms, sum of ops ${f1(x.sumOpsMs)} ms; ${parts}`);
  }
  finishRun(runId, { kept: 'diag', browser: r.browser, note: `bench/results/r8-diag-prof-${browser}-${short}.json; load ${r.loadStart}; ${r.results.map((x) => `L${x.len} ${f1(x.profile.totalMs)} ms`).join(' ')}` });
}

async function batch(browser, slug) {
  const m = MODELS.find((x) => x.slug === slug);
  const threads = val('--threads', '8');
  const runId = `r8-hc9-batch-${browser}-${m.short}-${Date.now()}`;
  appendRun(runId, { change: 'r8-hc9', commit, engine: 'kleinhirn', model: slug, bucket: 'real', precision: 'wasm-f32', browser,
    note: `runIdsBatch over the golden items, batch plans on/off/off/on, threads ${threads}` });
  const r = await page(browser, `${ctx.base}/bench/r8-batch.html?model=${slug}&task=${m.task}&threads=${threads}`, 'khR8Batch', runId);
  writeFileSync(join(RESULTS, `r8-hc9-batch-${browser}-${m.short}.json`), `${JSON.stringify({ date: new Date().toISOString(), commit, ...r }, null, 1)}\n`);
  const change = r.medianOn / r.medianOff - 1;
  console.log(`${r.browser} ${m.short}: batch plans ${r.medianOn.toFixed(2)} ms/item, rows one by one ${r.medianOff.toFixed(2)} (${(100 * change).toFixed(1)} %), bit-equal ${r.bitEqual}`);
  finishRun(runId, { kept: 'diag', browser: r.browser, median_ms: r.medianOn.toFixed(3),
    note: `bench/results/r8-hc9-batch-${browser}-${m.short}.json; on ${r.medianOn.toFixed(3)} off ${r.medianOff.toFixed(3)} ms/item; bit-equal ${r.bitEqual}` });
}

try {
  await startVite(ctx);
  for (const b of browsers) {
    if (mode === 'gemm') await gemm(b);
    else if (mode === 'prof') for (const m of models) await prof(b, m);
    else if (mode === 'batch') for (const m of models) await batch(b, m);
    else throw new Error('usage: run-r8-diag.mjs gemm|prof');
  }
} catch (e) {
  if (e instanceof PauseError) { console.error(`PAUSE: ${e.message}`); process.exitCode = 3; } else throw e;
} finally {
  stopVite(ctx);
}
