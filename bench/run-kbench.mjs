// R1 runner (docs/R1_WORKORDER.md): the kleinhirn bench table, kleinhirn against ONNX Runtime Web
// in its best setting per browser and way, on one Vite server with cross-origin isolation.
//
// Usage: node bench/run-kbench.mjs <probe|control|screen|table|mem|summary|--all>
//        [--browsers chromium,webkit,firefox,safari] [--ways f16,f32,wasm]
//        [--models minilm,roberta-base,mminilm,distilbert,deberta-v3-base,granite]
//        [--lengths 128,512,real] [--engines kleinhirn,ort]
// Stages in order of --all: probe (what starts per browser), control (Festlegung 12, Chromium f16
// full length against FINDINGS §43, hard stop above 5 %), screen (ORT best setting, writes
// data/kbench/ort-best.json), table (kleinhirn and ORT per cell, alternating order), mem (memory
// peak per cell, Chromium and Firefox), summary (bench/results/r1-table.json and r1-table.md).
//
// Every run: a fresh browser, the 1-minute load below 4 at the start (wait up to 30 minutes, then
// pause: exit 3), no foreign bench process, one runs.tsv line before the run (change r1-probe,
// r1-control, r1-screen, r1-table, r1-mem; the probe writes one line per browser), a result JSON
// bench/results/r1-<stage>-<browser>-<engine>-<model>-<length>-<way>-<stamp>.json. State:
// data/kbench/state.json (atomic); run the same command again to resume.
// Stage official (README cells, Noa 06.10.): kleinhirn against ORT in its best setting at real
// length, Chromium and Safari, f16 and f32, six models (24 cells; filters narrow them), order
// A B B A A B per cell (K28.8 Festlegung 12), three runs per engine. Committed code only:
// ../kleinhirn-official is checked out at this worktree's HEAD, built and served from there, and
// set back to its own HEAD at the end. Result <tag>-official-summary.json and .md.
// Table option --ort-threads <n> (R2): WASM cells run ORT a second time with numThreads n (engine key ort-t<n>);
// stage wasm-summary writes the WASM cells with both ORT runs to <tag>-table.json and .md.
// Stage wasm-probe (R2 stage 0): the env page with its WASM fields per browser, bench/results/<tag>-probe.json
// (KBENCH_TAG=r2: R2 stage 0; r8: firefox-reg, whose entry the table and screen stages read).
// Stage ab (R1b, docs/R2_WORKORDER.md): kleinhirn of this worktree against an older build copied to
// dist-ref/ (engine kleinhirn-ref, tools/r8_ref.mjs), ABAB per cell in one session, result <tag>-ab.json.
// WASM cells (R8, docs/R8_WORKORDER.md Festlegung 1): sample kind ab, parity per run from the
// outputs against the CPU f32 reference (no golden pass on the page), --threads <n|auto> for the
// new build and --threads-ref <n> for dist-ref/; committed engine files only.
// KBENCH_TAG (default r1) prefixes run ids, result files and the runs.tsv change.
// Exit codes: 3 pause (load or foreign processes), 4 control missed 5 %, 5 kleinhirn parity.

import { execFileSync, spawn } from 'node:child_process';
import { existsSync, lstatSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs';
import { loadavg } from 'node:os';
import { join, resolve } from 'node:path';
import { Worker } from 'node:worker_threads';
import {
  BROWSERS, ENGINES, EXTRA_BROWSERS, MODELS, PauseError, RESULTS, ROOT, STATE_FILE, WAYS,
  appendRun, blockForTimerStep, buildIdOnDisk, bundleBytes, cellKey, cellMatrix, checkEnvironment, checkServedBuild,
  compareToReference, engineGzip, engineOrder, finishRun, geoChange, killStalePid, median, missingReason,
  modelOf, openBrowser, parseFilters, parityVerdict, readJson, renderMarkdown, serveFrom, servedRoot, sh, sleep,
  startVite, stopVite, summarizeTable, writeJsonAtomic,
} from './kbench/runner-lib.mjs';

const BEST_FILE = join(ROOT, 'data/kbench/ort-best.json');
const K288_BEST = join(ROOT, 'data/k28/k28.8-best.json');
const OFFICIAL_SUMMARY = join(RESULTS, 'k28.8-official-summary.json');
const PROBE_FILE = join(RESULTS, 'r1-probe.json');
const MB = 1048576;

class StopError extends Error { constructor(code, msg) { super(msg); this.code = code; } }
class AutomationError extends Error {}

const ctx = { ownPids: new Set(), vite: null, base: null, commit: null };
let state = readJson(STATE_FILE, {});
state.runs ??= {};
state.automation ??= {};
const save = () => { mkdirSync(join(ROOT, 'data/kbench'), { recursive: true }); writeJsonAtomic(STATE_FILE, state); };

// ---------------------------------------------------------------- helpers

const TAG = process.env.KBENCH_TAG ?? 'r1';
const isKh = (engine) => engine.startsWith('kleinhirn');

// KBENCH_FF_WASM=fast (R2, Noa 07.10.): Firefox-WASM cells with warmup 1 and n = 3 at full length,
// 20 golden items at real length, no golden recomputation on the page (acc off) and the kleinhirn
// outputs kept: tools/r2_bitcheck.ts compares them bit for bit with Chromium and Node.
const FF_FAST = process.env.KBENCH_FF_WASM === 'fast';
const ffFast = (c) => FF_FAST && c.browser === 'firefox' && c.way === 'wasm';

const sampleSizes = (way, len, kind, browser = 'chromium') => {
  const wasm = way === 'wasm';
  const real = len === 'real';
  // Firefox ORT-wasm is about 16 times slower than Chromium-wasm (r1-probe.json): shorter samples
  // (Noa 05.10.): warmup 2, n = 10 at full length, the first 50 golden items at real length. The
  // table marks these cells.
  if (wasm && browser === 'firefox' && kind !== 'control') return FF_FAST ? { n: real ? 20 : 3, warmup: 1 } : { n: real ? 50 : 10, warmup: 2 };
  if (kind === 'screen') return wasm ? { n: real ? 40 : 20, warmup: 3 } : { n: 100, warmup: 20 };
  if (kind === 'mem') return wasm ? { n: real ? 40 : 10, warmup: 2 } : { n: real ? 60 : 40, warmup: 10 };
  if (kind === 'control') return { n: 300, warmup: 20 };
  // R8 step (Festlegung 1): Chromium n = 50, WebKit and the others n = 20, full length only
  if (kind === 'ab' && wasm) return browser === 'chromium' ? { n: 50, warmup: 5 } : { n: 20, warmup: 3 };
  return wasm ? { n: real ? 200 : 50, warmup: 5 } : { n: real ? 200 : 100, warmup: 20 };
};

const ortLabel = (c) => (c ? `${c.graph ?? 'std'}-${c.build ?? 'jsep'}-c${c.capture ? 1 : 0}${c.dynamic ? '-dyn' : ''}${c.threads !== undefined && c.threads !== null ? `-t${c.threads}` : ''}` : '');

// Memory pass: the page keeps computing this long after the measured calls (kbench.ts hold).
const MEM_HOLD_MS = 10000;

function pageUrl(spec) {
  const q = new URLSearchParams();
  if (spec.mode === 'env') { q.set('mode', 'env'); return `${ctx.base}/bench/kbench.html?${q}`; }
  const m = modelOf(spec.slug);
  q.set('engine', spec.engine); q.set('model', spec.slug); q.set('task', m.task); q.set('way', spec.way);
  q.set('len', spec.len); q.set('n', String(spec.n)); q.set('warmup', String(spec.warmup));
  if (!spec.acc) q.set('acc', '0');
  if (spec.mem) q.set('hold', String(MEM_HOLD_MS));
  if (spec.engine === 'ort') {
    const c = spec.cfg ?? {};
    q.set('graph', c.graph ?? 'std'); q.set('build', c.build ?? 'jsep'); q.set('capture', c.capture ? '1' : '0');
    q.set('dynamic', c.dynamic ? '1' : '0'); q.set('threads', String(c.threads ?? 1));
  } else if (spec.threads !== undefined && spec.threads !== null) {
    q.set('threads', String(spec.threads));
  }
  if (spec.split) q.set('split', spec.split);
  return `${ctx.base}/bench/kbench.html?${q}`;
}

const deltaSim = (slug) => readJson(join(ROOT, 'bench/results/k28-f16-sim', `${slug}.json`), {}).delta_sim ?? null;

function saveOutputs(slug, runId, outputs) {
  const dir = join(ROOT, 'models/k28', slug, 'k28.8/outputs');
  mkdirSync(dir, { recursive: true });
  const buf = Buffer.from(outputs.base64, 'base64');
  const file = join(dir, `${runId}.bin`);
  writeFileSync(file, buf);
  return { file: file.replace(`${ROOT}/`, ''), rows: outputs.rows, width: outputs.width, nonFinite: outputs.nonFinite, firstInput: outputs.firstInput };
}

// Memory: a worker thread samples the footprint of the runner's process tree every second
// (bench/mem-worker.mjs, method of run-official.mjs).
function startSampler() {
  const samples = [];
  const worker = new Worker(new URL('./mem-worker.mjs', import.meta.url),
    { workerData: { metric: 'footprint', rootPid: process.pid, intervalMs: 1000 } });
  worker.on('message', (s) => samples.push(s));
  worker.on('error', () => {});
  return { samples, stop: () => worker.terminate() };
}

// spec: { stage, browser, engine, slug, way, len, n, warmup, cfg, acc, mem, mode, tsv, note, keepOutputs }
async function runPage(spec) {
  const env = await checkEnvironment(ctx);
  const short = spec.mode === 'env' ? 'env' : modelOf(spec.slug).short.replace(/[^A-Za-z0-9-]/g, '');
  const runId = `${TAG}-${spec.stage}-${spec.browser}-${spec.mode === 'env' ? 'env' : spec.engine}-${short}-${spec.len ?? 'x'}-${spec.way ?? 'x'}${spec.cfg ? `-${ortLabel(spec.cfg)}` : ''}-${Date.now()}`;
  const stamp = runId.slice(TAG.length + 1);
  const file = join(RESULTS, `${TAG}-${stamp}.json`);
  const precision = spec.way === 'wasm' ? 'wasm-f32' : (spec.way ?? '').replace('webgpu-', '');
  if (spec.tsv !== false && spec.mode !== 'env') {
    appendRun(runId, {
      change: `${TAG}-${spec.stage}`, commit: ctx.commit, engine: isKh(spec.engine) ? spec.engine : 'ort-web', model: spec.slug,
      bucket: spec.len === 'real' ? 'real' : `L${spec.len}`, precision, browser: spec.browser,
      note: `${spec.engine === 'ort' ? `${ortLabel(spec.cfg)} ` : ''}n ${spec.n} warmup ${spec.warmup}${spec.note ? ` ${spec.note}` : ''}`,
    });
  }
  const fail = (error) => {
    if (spec.tsv !== false && spec.mode !== 'env') finishRun(runId, { kept: 'error', note: `${spec.browser} ${error}`.slice(0, 300) });
    return { runId, file: null, error: String(error).slice(0, 400) };
  };
  if (spec.engine === 'kleinhirn') await checkServedBuild(ctx);
  if (spec.engine === 'kleinhirn-ref' && !existsSync(join(ROOT, 'dist-ref/kleinhirn.js'))) throw new Error('dist-ref/kleinhirn.js missing');
  let drv;
  let sampler = null;
  let baselineMb = null;
  try {
    try {
      drv = await openBrowser(spec.browser);
    } catch (e) {
      throw new AutomationError(`${spec.browser}: ${String(e.message ?? e).slice(0, 200)}`);
    }
    if (spec.mem) {
      sampler = startSampler();
      await sleep(2500);
      const base = sampler.samples.map((s) => s.bytes).filter(Number.isFinite);
      baselineMb = base.length ? median(base) / MB : null;
    }
    const preCount = sampler?.samples.length ?? 0;
    try {
      await drv.goto(pageUrl(spec));
    } catch (e) {
      throw new AutomationError(`${spec.browser} goto: ${String(e.message ?? e).slice(0, 200)}`);
    }
    const t0 = Date.now();
    const pollMs = spec.browser === 'safari' ? 2000 : 500;
    let st;
    while (Date.now() - t0 < 40 * 60000) {
      st = await drv.state();
      if (st.done) break;
      await sleep(pollMs);
    }
    if (!st?.done) return fail(`page timeout (${st?.stage})`);
    if (st.error) return fail(st.error);
    const result = await drv.result();
    const memSamples = sampler ? sampler.samples.map((s) => s.bytes).filter(Number.isFinite) : [];
    const doc = { run_id: runId, change: `r1-${spec.stage}`, commit: ctx.commit, date: new Date().toISOString(), spec, browser: drv.label,
      loadStart: env.loadStart, loadWaitedSeconds: env.waitedSeconds, loadEnd: loadavg().map((v) => v.toFixed(2)).join(' '), ...result };
    if (sampler) {
      const peak = memSamples.length ? Math.max(...memSamples) / MB : null;
      const baseSamples = memSamples.slice(0, Math.max(preCount, 1));
      const base = baseSamples.length ? median(baseSamples) / MB : baselineMb;
      doc.mem = { metric: 'footprint', intervalMs: 1000, samples: memSamples.length, baselineMb: base, peakMb: peak, peakMinusBaselineMb: peak !== null && base !== null ? peak - base : null };
    }
    if (spec.mode === 'env') {
      writeFileSync(file, JSON.stringify(doc, null, 1));
      return { runId, file, env: result.env ?? null, crossOriginIsolated: result.crossOriginIsolated, timerStepMs: result.timerStepMs, browser: drv.label, error: result.error ?? null };
    }
    let outputs = null;
    if (result.outputs && spec.keepOutputs) outputs = saveOutputs(spec.slug, runId, result.outputs);
    let vsRef = null;
    if (result.outputs && spec.len !== 'real' && !result.error) {
      const ref = join(ROOT, 'models/k28', spec.slug, `k28.8/cpu-f32-L${spec.len}.bin`);
      if (existsSync(ref)) vsRef = compareToReference(result.outputs, readFileSync(ref), result.outputs.firstInput, result.outputs.width, modelOf(spec.slug).task);
    }
    const sizes = result.outputs ? { rows: result.outputs.rows, nonFinite: result.outputs.nonFinite } : null;
    delete doc.outputs;
    doc.outputs = outputs ?? sizes;
    doc.vsCpuF32 = vsRef;
    if (result.error) {
      writeFileSync(file, JSON.stringify(doc, null, 1));
      return { ...fail(result.error), file };
    }
    const gz = engineGzip(servedRoot(), spec.engine, result.engineFiles);
    let parity = { report: { accuracy: result.accuracy, vsCpuF32: vsRef } };
    if (isKh(spec.engine) && spec.acc !== false) {
      parity = { ...parityVerdict(spec.way, modelOf(spec.slug).task, result.accuracy, vsRef, deltaSim(spec.slug), modelOf(spec.slug).normalized === true), accuracy: result.accuracy, vsCpuF32: vsRef };
    } else if (isKh(spec.engine) && spec.parityFromRef) {
      // R8 ab: the verdict from the CPU f32 reference alone; no reference is a failure
      const v = vsRef ? parityVerdict(spec.way, modelOf(spec.slug).task, {}, vsRef, deltaSim(spec.slug), modelOf(spec.slug).normalized === true)
        : { pass: false, reasons: ['keine CPU-f32-Referenz'] };
      parity = { ...v, vsCpuF32: vsRef };
    }
    const run = {
      runId, file: file.replace(`${ROOT}/`, ''), engine: spec.engine, browser: drv.label,
      medianMs: result.latency.medianMs, p95Ms: result.latency.p95Ms, block: result.block, n: result.n,
      downloadMb: result.download ? result.download.bytes / MB : null, loadMs: result.loadMs, gpuBytes: result.info?.gpuBytes ?? null,
      engineGzipB: gz.bytes, config: spec.engine === 'ort' ? { ...spec.cfg } : { precision: spec.way,
        ...(spec.way === 'wasm' ? { threads: result.info?.config?.threads ?? null, threadNote: result.info?.config?.threadNote ?? null,
          wasmBuild: result.info?.config?.wasmBuild ?? null } : {}) }, parity,
      timerStepMs: result.timerStepMs, isolated: result.crossOriginIsolated, loadStart: env.loadStart,
      peakMemMib: doc.mem?.peakMinusBaselineMb ?? null, memSamples: doc.mem?.samples ?? null, error: null,
    };
    doc.summary = run;
    writeFileSync(file, JSON.stringify(doc, null, 1));
    const acc = result.accuracy ?? {};
    if (spec.tsv !== false) finishRun(runId, {
      argmax_agreement: typeof (acc.argmaxAgreement ?? acc.bestPassageAgreement) === 'number'
        ? (acc.argmaxAgreement ?? acc.bestPassageAgreement).toFixed(4) : (typeof acc.minCosine === 'number' ? acc.minCosine.toFixed(6) : ''),
      max_abs_logit_diff: typeof (acc.maxAbsLogitDiff ?? acc.maxAbsDiff) === 'number' ? (acc.maxAbsLogitDiff ?? acc.maxAbsDiff).toExponential(2) : '',
      median_ms: result.latency.medianMs.toFixed(3), p95_ms: result.latency.p95Ms.toFixed(3),
      model_only_median_ms: result.latency.medianMs.toFixed(3), load_ms: Math.round(result.loadMs),
      download_mb: run.downloadMb ? run.downloadMb.toFixed(1) : '', gpu_mb: run.gpuBytes ? (run.gpuBytes / MB).toFixed(1) : '',
      peak_mem_mb: run.peakMemMib ? run.peakMemMib.toFixed(0) : '', browser: drv.label,
      adapter: JSON.stringify(result.info?.adapter ?? {}), kept: '',
      note: `${run.file}; load ${env.loadStart}; isolated ${result.crossOriginIsolated} step ${result.timerStepMs} block ${result.block}`,
    });
    console.log(`${runId}: median ${result.latency.medianMs.toFixed(3)} ms p95 ${result.latency.p95Ms.toFixed(3)} (load ${env.loadStart}${result.block > 1 ? ', block 10' : ''})`);
    return { ...run, doc };
  } finally {
    if (sampler) await sampler.stop();
    if (drv) { try { await drv.close(); } catch { /* ignore */ } }
  }
}

// Run with a state record: skip a finished key, count automation failures per browser (two = missing).
async function recorded(key, spec) {
  const done = state.runs[key];
  if (done) return done;
  const auto = state.automation[spec.browser];
  if (auto?.failed) throw new AutomationError(auto.failed);
  let rec;
  try {
    rec = await runPage(spec);
  } catch (e) {
    if (!(e instanceof AutomationError)) throw e;
    const a = (state.automation[spec.browser] ??= { fails: 0 });
    a.fails += 1;
    a.last = e.message;
    if (a.fails >= 2) a.failed = e.message;
    save();
    throw e;
  }
  const slim = { ...rec };
  delete slim.doc;
  state.runs[key] = slim;
  save();
  return slim;
}

// ---------------------------------------------------------------- ORT configuration

const k288 = () => readJson(K288_BEST, { cells: {} });
const graphType = (slug) => k288().cells[`${slug}|L128`]?.winner?.graph ?? 'std';
const ortBest = () => readJson(BEST_FILE, { cells: {} });
const onnx = (slug, name) => existsSync(join(ROOT, 'models/k28', slug, 'k28.8/onnx', name));
const exportReport = (slug) => readJson(join(RESULTS, `r1-export-${slug}.json`), { graphs: {} });

function dynamicOk(slug, way) {
  const t = graphType(slug);
  const g = exportReport(slug).graphs[`Ldyn-${t}`];
  const prec = way === 'webgpu-f16' ? 'f16' : 'f32';
  return Boolean(g?.admissibleCpu) && onnx(slug, `Ldyn-${t}-${prec}.onnx`);
}

function fixedOk(slug, way, L) {
  const prec = way === 'webgpu-f16' ? 'f16' : 'f32';
  return onnx(slug, `L${L}-${graphType(slug)}-${prec}.onnx`);
}

// firefox-reg (R8): its entry of r8-probe.json (stage wasm-probe with KBENCH_TAG=r8); a probe error
// marks the browser as missing.
const probeOf = (browser) => {
  if (!EXTRA_BROWSERS.includes(browser)) return readJson(PROBE_FILE, { browsers: {} }).browsers[browser] ?? null;
  const e = readJson(join(RESULTS, 'r8-probe.json'), { browsers: {} }).browsers[browser];
  return e ? { ...e, ...(e.error ? { automationError: e.error } : {}) } : null;
};

// ---------------------------------------------------------------- stage: probe

async function stageProbe(f) {
  const slug = MODELS[0].slug;
  const all = readJson(PROBE_FILE, { browsers: {} });
  for (const browser of f.browsers) {
    if (state.probe?.[browser]) continue;
    console.log(`probe ${browser}`);
    // One runs.tsv line per browser, also across load pauses and resumes.
    state.probeIds ??= {};
    const fresh = !state.probeIds[browser];
    const probeId = state.probeIds[browser] ?? `r1-probe-${browser}-${Date.now()}`;
    state.probeIds[browser] = probeId;
    save();
    if (fresh) appendRun(probeId, { change: 'r1-probe', commit: ctx.commit, engine: 'probe', model: slug, bucket: 'L128', browser, note: 'env, kleinhirn f16 and f32, ORT builds x capture, ORT wasm threads, dynamic graph' });
    const entry = { browser, date: new Date().toISOString(), capability: { kleinhirn: {}, ort: {} }, runs: {} };
    const base = { stage: 'probe', browser, slug, tsv: false, acc: false, n: 10, warmup: 3 };
    try {
      let env;
      for (let attempt = 0; attempt < 2; attempt += 1) { // Safari automation: two failures make the cells missing
        try { env = await recorded(`probe|${browser}|env`, { stage: 'probe', browser, mode: 'env', tsv: false }); break; } catch (e) { if (!(e instanceof AutomationError) || attempt === 1) throw e; }
      }
      if (env.error) throw new AutomationError(`env page: ${env.error}`);
      entry.label = env.browser;
      entry.env = env.env;
      entry.crossOriginIsolated = env.crossOriginIsolated ?? null;
      entry.timerStepMs = env.timerStepMs ?? null;
      entry.blockSize = blockForTimerStep(entry.timerStepMs ?? 1);
      const hasGpu = Boolean(entry.env?.webgpu && entry.env?.adapter);
      const f16 = Boolean(entry.env?.adapter?.shaderF16);
      const probeRun = async (label, spec) => {
        const r = await recorded(`probe|${browser}|${label}`, { ...base, ...spec });
        entry.runs[label] = r.error ? { ok: false, error: r.error } : { ok: true, medianMs: r.medianMs, block: r.block, warmupOnly: undefined };
        return entry.runs[label];
      };
      for (const way of ['webgpu-f16', 'webgpu-f32']) {
        if (!hasGpu || (way === 'webgpu-f16' && !f16)) { entry.capability.kleinhirn[way] = { startable: false, error: !hasGpu ? 'kein WebGPU' : 'kein shader-f16' }; continue; }
        const r = await probeRun(`kleinhirn-${way}`, { engine: 'kleinhirn', way, len: '128' });
        entry.capability.kleinhirn[way] = { startable: r.ok, error: r.error ?? null };
      }
      const gpuWay = f16 ? 'webgpu-f16' : 'webgpu-f32';
      for (const way of ['webgpu-f16', 'webgpu-f32']) {
        const builds = {};
        if (!hasGpu || (way === 'webgpu-f16' && !f16)) { entry.capability.ort[way] = { startable: false, error: !hasGpu ? 'kein WebGPU' : 'kein shader-f16', builds }; continue; }
        for (const build of ['jsep', 'webgpu', 'jspi']) {
          builds[build] = {};
          for (const capture of [false, true]) {
            const r = await probeRun(`ort-${way}-${build}-c${capture ? 1 : 0}`, { engine: 'ort', way, len: '128', cfg: { graph: 'std', build, capture } });
            builds[build][capture ? 'capture' : 'plain'] = r.ok ? { ok: true, medianMs: r.medianMs } : { ok: false, error: r.error };
          }
        }
        const any = Object.values(builds).some((b) => b.plain.ok || b.capture.ok);
        entry.capability.ort[way] = { startable: any, error: any ? null : Object.values(builds).map((b) => b.plain.error).filter(Boolean)[0] ?? 'kein Build startet', builds };
      }
      const wasm = {};
      for (const threads of [1, 'hc']) {
        const r = await probeRun(`ort-wasm-t${threads}`, { engine: 'ort', way: 'wasm', len: '128', n: 5, warmup: 2, cfg: { graph: 'std', build: 'jsep', capture: false, threads } });
        wasm[`t${threads}`] = r;
      }
      entry.capability.ort.wasm = { startable: Object.values(wasm).some((w) => w.ok), error: Object.values(wasm).map((w) => w.error).filter(Boolean)[0] ?? null, threads: wasm };
      // Dynamic graph: smoke run at the real length (first working non-capture build).
      const dyn = {};
      const okBuild = ['webgpu', 'jspi', 'jsep'].find((b) => entry.capability.ort[gpuWay]?.builds?.[b]?.plain.ok);
      if (okBuild) dyn[gpuWay] = await probeRun(`ort-dyn-${gpuWay}-${okBuild}`, { engine: 'ort', way: gpuWay, len: 'real', n: 20, cfg: { graph: 'std', build: okBuild, capture: false, dynamic: true } });
      dyn.wasm = await probeRun('ort-dyn-wasm', { engine: 'ort', way: 'wasm', len: 'real', n: 10, warmup: 2, cfg: { graph: 'std', build: 'jsep', capture: false, dynamic: true, threads: 1 } });
      entry.capability.ort.dynamic = dyn;
      finishRun(probeId, { kept: '', note: `${entry.label}; webgpu ${hasGpu} shader-f16 ${f16} isolated ${entry.crossOriginIsolated} step ${entry.timerStepMs} block ${entry.blockSize}` });
    } catch (e) {
      if (!(e instanceof AutomationError)) throw e;
      entry.automationError = e.message;
      finishRun(probeId, { kept: 'error', note: e.message.slice(0, 250) });
    }
    all.browsers[browser] = entry;
    all.date = new Date().toISOString();
    all.commit = ctx.commit;
    writeJsonAtomic(PROBE_FILE, all);
    (state.probe ??= {})[browser] = true;
    save();
  }
}

// ---------------------------------------------------------------- ORT best setting

function k288Winner(slug, L) {
  const w = k288().cells[`${slug}|L${L}`]?.winner;
  return w ? { graph: w.graph, build: w.build, capture: w.capture === true } : null;
}

function builds(browser, way) {
  const cap = probeOf(browser)?.capability?.ort?.[way]?.builds ?? {};
  const out = [];
  for (const [build, v] of Object.entries(cap)) {
    if (v.plain?.ok) out.push({ build, capture: false });
    if (v.capture?.ok) out.push({ build, capture: true });
  }
  return out;
}

// One screening candidate; returns { cfg, median, error }.
async function screenRun(browser, way, slug, len, cfg) {
  const sz = sampleSizes(way, len, 'screen', browser);
  const key = `screen|${browser}|${way}|${slug}|${len}|${ortLabel(cfg)}`;
  const r = await recorded(key, { stage: 'screen', browser, engine: 'ort', slug, way, len, cfg, acc: false, ...sz });
  return { cfg, median: r.error ? null : r.medianMs, error: r.error ?? null, runId: r.runId, block: r.block };
}

const pickWinner = (cands) => cands.filter((c) => c.median !== null).sort((a, b) => a.median - b.median)[0] ?? null;

async function stageScreen(f) {
  const best = ortBest();
  best.workorder = 'docs/R1_WORKORDER.md, Festlegung 7 und 8';
  best.rule = 'winner: smallest screening median per browser, way, model; L512 takes the L128 winner (Chromium f16: the K28.8 winners); real length separately (bucketed graphs against the dynamic graph)';
  for (const browser of f.browsers) {
    const p = probeOf(browser);
    for (const way of f.ways) {
      if (missingReason(browser, way, 'ort', p)) continue;
      for (const slug of f.models) {
        const type = graphType(slug);
        const short = modelOf(slug).short;
        const prefix = `${browser}|${way}|${slug}`;
        const wasm = way === 'wasm';
        // L128
        let w128 = null;
        if (f.lengths.some((l) => l === '128' || l === '512' || l === 'real')) {
          if (!wasm && browser === 'chromium' && way === 'webgpu-f16') {
            const s = k288().cells[`${slug}|L128`];
            w128 = { cfg: k288Winner(slug, 128), median: s?.winner?.screeningMedian ?? null, source: 'k28.8-best.json' };
          } else if (fixedOk(slug, way, 128)) {
            const list = wasm ? [1, 'hc'].map((threads) => ({ graph: type, build: 'jsep', capture: false, threads }))
              : builds(browser, way).map((b) => ({ graph: type, ...b }));
            const cands = [];
            for (const cfg of list) cands.push(await screenRun(browser, way, slug, '128', cfg));
            const w = pickWinner(cands);
            best.cells[`${prefix}|L128`] = { slug, short, browser, way, L: 128, family: modelOf(slug).family, task: modelOf(slug).task,
              winner: w ? { ...w.cfg, screeningMedian: w.median, runs: 1 } : null, candidates: cands };
            w128 = w ? { cfg: w.cfg, median: w.median, source: 'r1 screening' } : null;
          }
          if (!best.cells[`${prefix}|L128`] && w128) {
            best.cells[`${prefix}|L128`] = { slug, short, browser, way, L: 128, winner: { ...w128.cfg, screeningMedian: w128.median, runs: 1, source: w128.source }, candidates: [] };
          }
        }
        // L512: the L128 winner, Chromium f16 the K28.8 winner of L512
        const w512cfg = !wasm && browser === 'chromium' && way === 'webgpu-f16' ? k288Winner(slug, 512) : w128?.cfg ?? null;
        if (w512cfg) {
          best.cells[`${prefix}|L512`] = { slug, short, browser, way, L: 512,
            winner: { ...w512cfg, source: browser === 'chromium' && way === 'webgpu-f16' ? 'k28.8-best.json' : 'L128 winner' }, candidates: [] };
        }
        // real length: bucketed graphs with the L128 winner, against the dynamic graph
        if (w128?.cfg && f.lengths.includes('real')) {
          const cands = [];
          if (fixedOk(slug, way, 128) && fixedOk(slug, way, 512)) {
            cands.push(await screenRun(browser, way, slug, 'real', w128.cfg));
            if (wasm) cands.push(await screenRun(browser, way, slug, 'real', { ...w128.cfg, threads: w128.cfg.threads === 1 ? 'hc' : 1 }));
          }
          if (dynamicOk(slug, way)) {
            const dynBuild = wasm ? 'jsep' : (probeOf(browser)?.capability?.ort?.dynamic?.[way]?.ok === false ? null : w128.cfg.build);
            if (dynBuild) {
              cands.push(await screenRun(browser, way, slug, 'real', { graph: type, build: dynBuild, capture: false, dynamic: true, ...(wasm ? { threads: w128.cfg.threads } : {}) }));
              if (wasm) cands.push(await screenRun(browser, way, slug, 'real', { graph: type, build: dynBuild, capture: false, dynamic: true, threads: w128.cfg.threads === 1 ? 'hc' : 1 }));
            }
          }
          const w = pickWinner(cands);
          best.cells[`${prefix}|real`] = { slug, short, browser, way, L: 'real', winner: w ? { ...w.cfg, screeningMedian: w.median, runs: 1 } : null, candidates: cands };
        }
        best.commit = ctx.commit;
        best.date = new Date().toISOString();
        mkdirSync(join(ROOT, 'data/kbench'), { recursive: true });
        writeJsonAtomic(BEST_FILE, best);
      }
    }
  }
}

function ortCfgFor(browser, way, slug, len) {
  const c = ortBest().cells[`${browser}|${way}|${slug}|${len === 'real' ? 'real' : `L${len}`}`];
  if (!c?.winner) return null;
  const { graph, build, capture, dynamic, threads } = c.winner;
  return { graph, build, capture: capture === true, ...(dynamic ? { dynamic: true } : {}), ...(threads !== undefined ? { threads } : {}) };
}

// ---------------------------------------------------------------- stage: control (Festlegung 12)

async function stageControl(f) {
  // Festlegung 12 (rewritten 05.10.): the new page against the unchanged K28.8 pages in the same
  // session, at most 5 % per cell and engine. Runs missing in the state are measured; stored runs
  // are re-evaluated (the parity verdict is recomputed from the stored accuracy).
  const official = readJson(OFFICIAL_SUMMARY, null);
  const rows = [];
  let idx = 0;
  for (const L of ['128', '512']) {
    for (const m of f.models) {
      const row = { slug: m, short: modelOf(m).short, L: Number(L) };
      for (const engine of engineOrder(idx)) {
        const spec = { stage: 'control', browser: 'chromium', engine, slug: m, way: 'webgpu-f16', len: L, ...sampleSizes('webgpu-f16', L, 'control'), acc: true, keepOutputs: true,
          ...(engine === 'ort' ? { cfg: { ...k288Winner(m, Number(L)) } } : {}) };
        const r = await recorded(`control|chromium|${engine}|${m}|${L}`, spec);
        let parity = r.parity;
        if (!r.error && engine === 'kleinhirn' && r.file) {
          const doc = readJson(join(ROOT, r.file), null);
          if (doc?.accuracy) parity = parityVerdict('webgpu-f16', modelOf(m).task, doc.accuracy, doc.vsCpuF32, deltaSim(m), modelOf(m).normalized === true);
        }
        row[engine] = r.error ? { error: r.error } : { median: r.medianMs, p95: r.p95Ms, runId: r.runId, file: r.file, loadStart: r.loadStart, parity };
      }
      idx += 1;
      rows.push(row);
    }
  }
  const old = await controlOldRuns(f);
  for (const row of rows) {
    const o = old[`${row.slug}|L${row.L}`];
    const off = official?.cells?.[`${row.slug}|L${row.L}`];
    for (const e of ENGINES) {
      const oldMedian = e === 'kleinhirn' ? o?.kh?.median : o?.ort?.median;
      row[e].old = oldMedian ?? null;
      row[e].deviation = typeof row[e].median === 'number' && typeof oldMedian === 'number' ? row[e].median / oldMedian - 1 : null;
      if (off) row[e].official = median(off[e].runs.map((r) => r.median));
    }
  }
  const devs = rows.flatMap((r) => ENGINES.map((e) => r[e].deviation));
  const complete = devs.every((d) => typeof d === 'number');
  const maxDev = complete ? Math.max(...devs.map(Math.abs)) : null;
  const parityOk = rows.every((r) => r.kleinhirn.parity?.pass === true);
  const out = { workorder: 'docs/R1_WORKORDER.md Festlegung 12 (neu gefasst 05.10.)', date: new Date().toISOString(), commit: ctx.commit,
    reference: 'K28.8 pages bench/k28-latency.html and bench/k28-ort.html in the same session (data/kbench/state.json controlOld, bench/results/r1-controlold-*.json); official §43 medians listed for information',
    rule: 'Chromium f16 full length, both engines, median of the new page within 5 % of the old page per cell and engine',
    maxAbsDeviation: maxDev, parityPass: parityOk, pass: complete && maxDev <= 0.05, rows };
  writeFileSync(join(RESULTS, 'r1-control.json'), JSON.stringify(out, null, 1));
  console.log(`control: max |new/old - 1| ${maxDev === null ? 'n/a' : (maxDev * 100).toFixed(2)} % (${out.pass ? 'pass' : 'FAIL'}), kleinhirn parity ${parityOk ? 'pass' : 'FAIL'}`);
  if (!out.pass) throw new StopError(4, `control (Festlegung 12) missed 5 %: ${maxDev === null ? 'incomplete' : (maxDev * 100).toFixed(2)} %; see bench/results/r1-control.json`);
}

// ---------------------------------------------------------------- diagnostic: the K28.8 pages as control

// Not part of --all. The unchanged K28.8 pages (bench/k28-latency.html, bench/k28-ort.html) in the same
// Vite and the same session, Chromium f16 full length, n = 300: tells apart "the new page measures
// something else" from "the machine is slower than on the night of FINDINGS §43".
async function runLegacy(side, slug, L, w, tag) {
  const env = await checkEnvironment(ctx);
  const short = modelOf(slug).short.replace(/[^A-Za-z0-9-]/g, '');
  const runId = `r1-controlold-chromium-${side}-${short}-L${L}-${Date.now()}`;
  const file = join(RESULTS, `${runId}.json`);
  appendRun(runId, { change: 'r1-control-old', commit: ctx.commit, engine: side === 'kh' ? 'kleinhirn' : 'ort-web', model: slug, bucket: `L${L}`,
    precision: 'f16', browser: 'chromium', note: `K28.8 page, n 300${side === 'ort' ? ` ${w.graph} ${w.build} c${w.capture ? 1 : 0}` : ''}` });
  if (side === 'kh') await checkServedBuild(ctx);
  const url = side === 'kh' ? `/bench/k28-latency.html?model=${slug}&L=${L}&n=300`
    : `/bench/k28-ort.html?model=${slug}&L=${L}&graph=${w.graph}&build=${w.build}&capture=${w.capture ? 1 : 0}&n=300`;
  const global = side === 'kh' ? 'khK28LatencyResult' : 'khK28OrtResult';
  const drv = await openBrowser('chromium', global);
  try {
    await drv.goto(`${ctx.base}${url}`);
    const t0 = Date.now();
    let st;
    while (Date.now() - t0 < 20 * 60000) { st = await drv.state(); if (st.done) break; await sleep(500); }
    const result = await drv.result();
    if (result.outputs) delete result.outputs.base64;
    writeFileSync(file, JSON.stringify({ run_id: runId, commit: ctx.commit, loadStart: env.loadStart, ...result }, null, 1));
    if (result.error) { finishRun(runId, { kept: 'error', note: String(result.error).slice(0, 200) }); return { error: result.error }; }
    finishRun(runId, { median_ms: result.latency.medianMs.toFixed(3), p95_ms: result.latency.p95Ms.toFixed(3), browser: drv.label, kept: '',
      note: `${file.replace(`${ROOT}/`, '')}; load ${env.loadStart}` });
    console.log(`${runId}: median ${result.latency.medianMs.toFixed(3)} (load ${env.loadStart})`);
    return { median: result.latency.medianMs, p95: result.latency.p95Ms, file: file.replace(`${ROOT}/`, ''), loadStart: env.loadStart };
  } finally {
    await drv.close();
  }
}

async function controlOldRuns(f) {
  state.controlOld ??= {};
  const out = {};
  let idx = 0;
  for (const L of [128, 512]) {
    for (const slug of f.models) {
      const w = k288Winner(slug, L);
      const rec = (state.controlOld[`${slug}|L${L}`] ??= {});
      for (const side of (idx % 2 === 0 ? ['kh', 'ort'] : ['ort', 'kh'])) {
        if (!rec[side]) { rec[side] = await runLegacy(side, slug, L, w); save(); }
      }
      out[`${slug}|L${L}`] = rec;
      idx += 1;
    }
  }
  return out;
}

async function stageControlOld(f) {
  await controlOldRuns(f);
}

// ---------------------------------------------------------------- stage: table and mem

function runSpec(kind, c, engine) {
  const sz = sampleSizes(c.way, c.len, kind, c.browser);
  const fast = kind === 'table' && ffFast(c);
  const base = { stage: kind, browser: c.browser, engine, slug: c.slug, way: c.way, len: c.len, ...sz, acc: kind === 'table' && !fast,
    keepOutputs: kind === 'table' && (c.len !== 'real' || (fast && isKh(engine))), mem: kind === 'mem', ...(fast ? { note: 'ff-fast' } : {}) };
  if (engine === 'ort') base.cfg = ortCfgFor(c.browser, c.way, c.slug, c.len);
  return base;
}

function cellMissing(c, engine) {
  const m = missingReason(c.browser, c.way, engine, probeOf(c.browser));
  if (m) return m;
  if (engine === 'ort') {
    const cfg = ortCfgFor(c.browser, c.way, c.slug, c.len);
    if (!cfg) {
      const why = ortBest().cells[`${c.browser}|${c.way}|${c.slug}|${c.len === 'real' ? 'real' : `L${c.len}`}`];
      const errs = (why?.candidates ?? []).map((x) => x.error).filter(Boolean);
      return `fehlt: ORT ohne lauffähige Einstellung (${errs[0] ?? 'kein Graph oder Build'})`.slice(0, 220);
    }
  }
  return null;
}

async function stageTable(f) {
  const cells = cellMatrix(f);
  let i = 0;
  for (const c of cells) {
    for (const engine of engineOrder(i).filter((e) => f.engines.includes(e))) {
      if (cellMissing(c, engine)) continue;
      try {
        // R8: --threads <n|auto> for kleinhirn-WASM
        const kh = engine === 'kleinhirn' && c.way === 'wasm' && f.threads ? { threads: f.threads } : {};
        const r = await recorded(`${TAG === 'r1' ? '' : `${TAG}-`}table|${cellKey(c)}|${engine}`, { ...runSpec('table', c, engine), ...kh });
        // R2 (Festlegung 7): ORT-wasm once more with numThreads from --ort-threads, right after its best setting
        if (engine === 'ort' && f.ortThreads && c.way === 'wasm') {
          const spec = runSpec('table', c, 'ort');
          await recorded(`${TAG === 'r1' ? '' : `${TAG}-`}table|${cellKey(c)}|ort-t${f.ortThreads}`, { ...spec, cfg: { ...spec.cfg, threads: f.ortThreads } });
        }
        if (engine === 'kleinhirn' && !r.error && r.parity && r.parity.pass === false) {
          throw new StopError(5, `kleinhirn parity fails in ${cellKey(c)}: ${r.parity.reasons.join('; ')} (${r.file})`);
        }
      } catch (e) {
        if (e instanceof AutomationError) { console.log(`skip ${cellKey(c)}: ${e.message}`); continue; }
        throw e;
      }
    }
    i += 1;
  }
}

async function stageMem(f) {
  for (const c of cellMatrix(f)) {
    for (const engine of ENGINES) {
      if (cellMissing(c, engine)) continue;
      if (c.browser !== 'chromium' && c.browser !== 'firefox') continue;
      try {
        // R8: tagged keys (the untagged ones hold R1/R2), --threads for kleinhirn-WASM
        const kh = engine === 'kleinhirn' && c.way === 'wasm' && f.threads ? { threads: f.threads } : {};
        await recorded(`${TAG.startsWith('r8') ? `${TAG}-` : ''}mem|${cellKey(c)}|${engine}`, { ...runSpec('mem', c, engine), ...kh });
      } catch (e) {
        if (e instanceof AutomationError) { console.log(`skip ${cellKey(c)}: ${e.message}`); continue; }
        throw e;
      }
    }
  }
}

// ---------------------------------------------------------------- stage: summary

function stageSummary(f) {
  const cells = cellMatrix({ ...f, browsers: BROWSERS, ways: WAYS });
  const get = (c, engine) => state.runs[`table|${cellKey(c)}|${engine}`];
  const memNote = (c) => (c.browser === 'chromium' || c.browser === 'firefox' ? 'nicht gemessen'
    : c.browser === 'webkit' ? 'fehlt: WebKit-Prozesse (XPC) nicht eindeutig zuordenbar' : 'fehlt: Safari teilt Prozesse und Tabs mit anderen Fenstern');
  const mem = (c, engine) => {
    const r = state.runs[`mem|${cellKey(c)}|${engine}`];
    if (r && !r.error && typeof r.peakMemMib === 'number') return { peakMemMib: r.peakMemMib, memNote: null };
    return { peakMemMib: null, memNote: r?.error ? `fehlt: ${r.error.slice(0, 80)}` : memNote(c) };
  };
  const miss = (c, engine) => {
    const auto = state.automation[c.browser]?.failed;
    if (auto) return `fehlt: Automatisierung (${auto.slice(0, 120)})`;
    return cellMissing(c, engine);
  };
  const table = summarizeTable(cells, get, mem, miss);
  const gz = { kleinhirn: bundleBytes().gzip9, ortByBuild: {} };
  for (const c of cells) {
    const r = get(c, 'ort');
    if (r && !r.error && r.engineGzipB) gz.ortByBuild[r.config?.build ?? 'jsep'] = r.engineGzipB;
  }
  const meta = { date: new Date().toISOString().slice(0, 10), commit: ctx.commit,
    gzip: { kleinhirn: gz.kleinhirn, ort: Object.entries(gz.ortByBuild).map(([b, v]) => `${b} ${v}`).join(', ') || '?' } };
  const probe = readJson(PROBE_FILE, { browsers: {} });
  const doc = { workorder: 'docs/R1_WORKORDER.md', date: new Date().toISOString(), commit: ctx.commit, gzipB: gz,
    cellsPerEngine: cells.length, ok: Object.fromEntries(ENGINES.map((e) => [e, table.filter((r) => r[e].status === 'ok').length])),
    probe: Object.fromEntries(Object.entries(probe.browsers).map(([b, p]) => [b, { label: p.label, webgpu: p.env?.webgpu, shaderF16: p.env?.adapter?.shaderF16, isolated: p.crossOriginIsolated, timerStepMs: p.timerStepMs, blockSize: p.blockSize, automationError: p.automationError ?? null }])),
    table };
  writeFileSync(join(RESULTS, 'r1-table.json'), JSON.stringify(doc, null, 1));
  writeFileSync(join(RESULTS, 'r1-table.md'), renderMarkdown(table, meta));
  console.log(`summary: ${cells.length} cells per engine, ok kleinhirn ${doc.ok.kleinhirn}, ORT ${doc.ok.ort}`);
}

// ---------------------------------------------------------------- stage: wasm-probe (R2)

// R2 stage 0 (docs/R2_WORKORDER.md): per browser the env page with its WASM fields (SIMD, relaxed
// SIMD, isolation, SharedArrayBuffer, hardwareConcurrency, largest growing memory). One runs.tsv
// line per browser (change r2-probe), result bench/results/r2-probe.json.
async function stageWasmProbe(f) {
  const file = join(RESULTS, `${TAG}-probe.json`);
  const all = readJson(file, { workorder: TAG === 'r8' ? 'docs/R8_WORKORDER.md' : 'docs/R2_WORKORDER.md', browsers: {} });
  for (const browser of f.browsers) {
    const runId = `${TAG}-probe-${browser}-${Date.now()}`;
    appendRun(runId, { change: `${TAG}-probe`, commit: ctx.commit, engine: 'probe', browser, note: 'WASM SIMD, relaxed SIMD, isolation, SharedArrayBuffer, hardwareConcurrency, largest growing memory' });
    let env;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try { env = await runPage({ stage: 'wasm-probe', browser, mode: 'env', tsv: false }); break; } catch (e) {
        if (!(e instanceof AutomationError) || attempt === 1) { env = { error: String(e.message ?? e) }; break; }
      }
    }
    const wasm = env?.env?.wasm ?? null;
    all.browsers[browser] = { date: new Date().toISOString(), label: env?.browser ?? null, commit: ctx.commit, file: env?.file ?? null,
      ...(env?.error ? { error: env.error } : {}), wasm };
    all.date = new Date().toISOString();
    writeFileSync(file, `${JSON.stringify(all, null, 1)}\n`);
    finishRun(runId, { kept: env?.error ? 'error' : 'probe', note: env?.error ? env.error.slice(0, 200) : `simd ${wasm?.simd} relaxed ${wasm?.relaxedSimd} coi ${wasm?.crossOriginIsolated} sab ${wasm?.sharedArrayBuffer} cores ${wasm?.hardwareConcurrency} mem ${wasm?.memory?.maxMiB} MiB shared ${wasm?.sharedMemory?.maxMiB ?? '-'} MiB` });
    console.log(`wasm-probe ${browser}: ${env?.error ?? JSON.stringify(wasm)}`);
  }
}

// ---------------------------------------------------------------- stage: wasm-summary (R2)

// R2 table (docs/R2_WORKORDER.md Stufe 3): the WASM cells of all four browsers, kleinhirn on one
// thread against ORT-wasm in its best setting (all cores) and with one thread (--ort-threads 1),
// memory peak in Chromium. Runs of <tag>-table and mem; result <tag>-table.json and .md.
function stageWasmSummary(f) {
  const cells = cellMatrix({ ...f, browsers: [...BROWSERS, ...EXTRA_BROWSERS], ways: ['wasm'] })
    .filter((c) => !EXTRA_BROWSERS.includes(c.browser) || state.runs[`${TAG}-table|${cellKey(c)}|kleinhirn`]);
  const t = (c, engine) => state.runs[`${TAG}-table|${cellKey(c)}|${engine}`];
  const entry = (c, engine) => {
    const r = t(c, engine);
    if (!r) return { status: 'fehlt: nicht gelaufen' };
    if (r.error) return { status: `fehlt: ${r.error.slice(0, 160)}` };
    // R1/R2 memory passes have untagged keys, R8 ones carry the tag
    const m = state.runs[`${TAG.startsWith('r8') ? `${TAG}-` : ''}mem|${cellKey(c)}|${engine.startsWith('ort') ? 'ort' : engine}`];
    return { status: 'ok', medianMs: r.medianMs, p95Ms: r.p95Ms, n: r.n, parity: r.parity ?? null, config: r.config ?? null,
      peakMemMib: m && !m.error && typeof m.peakMemMib === 'number' ? m.peakMemMib : null, file: r.file ?? null };
  };
  const rows = cells.map((c) => {
    const k = entry(c, 'kleinhirn');
    const o = entry(c, 'ort');
    const o1 = entry(c, 'ort-t1');
    const f1 = (a, b) => (a.status === 'ok' && b.status === 'ok' ? a.medianMs / b.medianMs : null);
    return { ...c, short: modelOf(c.slug).short, kleinhirn: k, ort: o, ortT1: o1, factorBest: f1(o, k), factorT1: f1(o1, k) };
  });
  const full = rows.filter((r) => r.browser === 'chromium' && r.len !== 'real');
  const gate = {
    cells: full.length,
    kleinhirnAhead: full.filter((r) => r.factorT1 !== null && r.factorT1 > 1).length,
    parity: rows.filter((r) => r.kleinhirn.status === 'ok').every((r) => r.kleinhirn.parity?.pass !== false),
  };
  // Gate R8 (docs/R8_WORKORDER.md): Chromium ahead of ORT best in all 18 cells; WebKit and Safari
  // the geometric mean of ORT best / kleinhirn over 18 cells at least 1; Firefox reported.
  const geo = (rs) => (rs.length ? Math.exp(rs.reduce((a, r) => a + Math.log(r.factorBest), 0) / rs.length) : null);
  const per = (b) => rows.filter((r) => r.browser === b && r.factorBest !== null);
  const r8 = {
    chromium: { cells: per('chromium').length, ahead: per('chromium').filter((r) => r.factorBest > 1).length, geo: geo(per('chromium')) },
    webkit: { cells: per('webkit').length, geo: geo(per('webkit')) },
    safari: { cells: per('safari').length, geo: geo(per('safari')) },
    'firefox-reg': { cells: per('firefox-reg').length, geo: geo(per('firefox-reg')) },
  };
  r8.pass = r8.chromium.cells === 18 && r8.chromium.ahead === 18 && r8.webkit.cells === 18 && r8.webkit.geo >= 1
    && r8.safari.cells === 18 && r8.safari.geo >= 1;
  gate.r8 = r8;
  const doc = { workorder: TAG.startsWith('r8') ? 'docs/R8_WORKORDER.md' : 'docs/R2_WORKORDER.md', date: new Date().toISOString(), commit: ctx.commit, bundle: bundleBytes(), gate, rows };
  writeFileSync(join(RESULTS, `${TAG}-table.json`), JSON.stringify(doc, null, 1));
  const fmt = (v, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d).replace('.', ',') : '');
  const cell = (e) => (e.status === 'ok' ? fmt(e.medianMs) : e.status.replace(/\|/g, '/'));
  const lenLabel = { 128: 'L128 voll', 512: 'L512 voll', real: 'echte Länge' };
  const threadsUsed = [...new Set(rows.map((r) => r.kleinhirn.config?.threads).filter((x) => x !== undefined))].join(', ');
  const out = [`# ${TAG}: kleinhirn-WASM gegen ORT-wasm`, '',
    `Stand ${doc.date.slice(0, 10)}, Commit ${ctx.commit}. Quelle \`bench/results/${TAG}-table.json\`, erzeugt von \`KBENCH_TAG=${TAG} node bench/run-kbench.mjs wasm-summary\`. Median in ms je Aufruf; kleinhirn mit ${threadsUsed || 'einem'} Thread(s), ORT in der Einstellung aus \`data/kbench/ort-best.json\` (alle Kerne) und mit \`numThreads\` 1; Faktor ORT durch kleinhirn; Speicher-Spitze in MiB nur Chromium.`, '',
    TAG.includes('nocoi')
      ? `Ohne Cross-Origin-Isolation (KBENCH_NO_COI=1): kleinhirn und ORT können keine Threads starten und rechnen mit einem Thread. Faktor ORT durch kleinhirn im geometrischen Mittel Chromium ${fmt(r8.chromium.geo)}, Safari ${fmt(r8.safari.geo)}.`
      : TAG.startsWith('r8')
      ? `Gate R8: Chromium kleinhirn vor ORT best in ${r8.chromium.ahead} von ${r8.chromium.cells} Zellen; Faktor ORT best durch kleinhirn im geometrischen Mittel WebKit ${fmt(r8.webkit.geo)}, Safari ${fmt(r8.safari.geo)}, Firefox ${fmt(r8['firefox-reg'].geo)} (berichtet), Chromium ${fmt(r8.chromium.geo)}. ${r8.pass ? 'Erfüllt.' : 'Nicht erfüllt.'}`
      : `Gate Geschwindigkeit (Chromium, L128 und L512 voll): kleinhirn vor ORT mit einem Thread in ${gate.kleinhirnAhead} von ${gate.cells} Zellen.`, ''];
  let last = null;
  for (const r of rows) {
    if (r.browser !== last) {
      out.push(`## ${r.browser}`, '', '| Modell | Länge | kleinhirn | p95 | MiB | ORT best | ORT 1 Thread | Faktor best | Faktor 1 Thread | Parität kleinhirn |', '|---|---|---|---|---|---|---|---|---|---|');
      last = r.browser;
    }
    const k = r.kleinhirn;
    const parity = k.status === 'ok' ? (k.parity?.pass ? 'ok' : (k.parity?.reasons ?? ['offen']).join('; ')) : '';
    out.push(`| ${r.short} | ${lenLabel[r.len]} | ${cell(k)} | ${k.status === 'ok' ? fmt(k.p95Ms) : ''} | ${k.peakMemMib !== null && k.peakMemMib !== undefined ? fmt(k.peakMemMib, 0) : ''} | ${cell(r.ort)} | ${cell(r.ortT1)} | ${fmt(r.factorBest)} | ${fmt(r.factorT1)} | ${parity} |`);
    if (r === rows[rows.length - 1] || rows[rows.indexOf(r) + 1].browser !== r.browser) out.push('');
  }
  writeFileSync(join(RESULTS, `${TAG}-table.md`), out.join('\n'));
  console.log(`wasm-summary: ${rows.length} cells, kleinhirn ok ${rows.filter((r) => r.kleinhirn.status === 'ok').length}, gate ${gate.kleinhirnAhead}/${gate.cells}`);
}

// ---------------------------------------------------------------- stage: ab (two builds)

// R8 Festlegung 1: measured engine code is committed code.
function assertCommitted(stage) {
  const dirty = sh('git', ['-C', ROOT, 'status', '--porcelain', '--', ...RUNNER_FILES.filter((x) => x !== 'data/kbench/ort-best.json')]);
  if (dirty) throw new Error(`${stage}: uncommitted engine or runner files (R8 Festlegung 1):\n${dirty}`);
}

async function stageAb(f) {
  const cells = cellMatrix(f);
  if (cells.some((c) => c.way === 'wasm')) {
    if (cells.some((c) => c.way === 'wasm' && c.len === 'real')) throw new Error('ab: WASM cells at full length only (R8 Festlegung 1)');
    assertCommitted('ab');
  }
  const refFile = join(ROOT, 'dist-ref/REF');
  const refInfo = existsSync(refFile) ? JSON.parse(readFileSync(refFile, 'utf8')) : null;
  const rows = [];
  for (const [i, c] of cells.entries()) {
    const order = i % 2 === 0 ? ['kleinhirn-ref', 'kleinhirn'] : ['kleinhirn', 'kleinhirn-ref'];
    const runs = { kleinhirn: [], 'kleinhirn-ref': [] };
    const wasm = c.way === 'wasm';
    for (const rep of [0, 1]) {
      for (const engine of order) {
        const spec = wasm
          ? { ...runSpec('ab', c, engine), stage: 'ab', acc: false, parityFromRef: true,
            threads: engine === 'kleinhirn' ? f.threads : f.threadsRef }
          : { ...runSpec('table', c, engine), stage: 'ab' };
        const r = await recorded(`${TAG}-ab|${cellKey(c)}|${engine}|${rep}`, spec);
        if (r.error) throw new Error(`ab ${cellKey(c)} ${engine}: ${r.error}`);
        runs[engine].push(r);
      }
    }
    const med = (rs) => median(rs.map((r) => r.medianMs));
    const ok = (r) => (wasm ? r.parity?.pass === true : r.parity?.pass !== false);
    const row = { cell: cellKey(c), browser: c.browser, short: modelOf(c.slug).short, len: c.len, way: c.way,
      refMs: med(runs['kleinhirn-ref']), newMs: med(runs.kleinhirn),
      parityNew: runs.kleinhirn.every(ok), parityRef: runs['kleinhirn-ref'].every(ok),
      runIds: [...runs['kleinhirn-ref'], ...runs.kleinhirn].map((r) => r.runId) };
    row.change = row.newMs / row.refMs - 1;
    rows.push(row);
    console.log(`${row.cell}: ref ${row.refMs.toFixed(2)} new ${row.newMs.toFixed(2)} ms (${(100 * row.change).toFixed(2)} %) parity new ${row.parityNew}`);
  }
  const byBrowser = {};
  for (const b of [...new Set(rows.map((r) => r.browser))]) {
    const rs = rows.filter((r) => r.browser === b);
    byBrowser[b] = { cells: rs.length, geoChange: geoChange(rs), worst: Math.max(...rs.map((r) => r.change)) };
    console.log(`ab ${b}: geometric mean ${(100 * byBrowser[b].geoChange).toFixed(2)} %, worst cell ${(100 * byBrowser[b].worst).toFixed(2)} %`);
  }
  const doc = { workorder: cells.some((c) => c.way === 'wasm') ? 'docs/R8_WORKORDER.md' : 'docs/R2_WORKORDER.md', date: new Date().toISOString(), commit: ctx.commit,
    ref: 'dist-ref/kleinhirn.js', refCommit: refInfo?.commit ?? null, refBuildId: readFileSync(join(ROOT, 'dist-ref/kleinhirn.js'), 'utf8').match(/buildId\s*:\s*["']([a-z0-9]+)["']/)?.[1] ?? null,
    newBuildId: buildIdOnDisk(), threads: f.threads ?? null, threadsRef: f.threadsRef ?? null, rows, byBrowser,
    worst: rows.reduce((w, r) => (r.change > w ? r.change : w), -Infinity),
    parityNew: rows.every((r) => r.parityNew) };
  writeFileSync(join(RESULTS, `${TAG}-ab.json`), JSON.stringify(doc, null, 1));
  console.log(`ab: ${rows.length} cells, worst ${(100 * doc.worst).toFixed(2)} %, parity new ${doc.parityNew}`);
}

// ---------------------------------------------------------------- stage: threads-search (R8 hc8)

// R8 hc8 (docs/R8_WORKORDER.md): kleinhirn-WASM per cell with every thread count of --thread-list
// and every work split of --splits ("rb,cb,qb;..."), one page run each, full length with the ab
// sample, real length with the table sample. One runs.tsv line per page run; result
// <tag>-search.json with the median per setting and the best per cell.
async function stageThreadsSearch(f, args) {
  const val = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
  const threadList = val('--thread-list', '1,2,4,6,8,10').split(',').map(Number);
  const splits = val('--splits', '').split(';').filter(Boolean);
  assertCommitted('threads-search');
  const rows = [];
  for (const c of cellMatrix(f).filter((x) => x.way === 'wasm')) {
    const cores = c.browser === 'webkit' || c.browser === 'safari' ? 8 : 10;
    for (const threads of threadList.filter((t) => t <= cores)) {
      for (const split of splits.length ? splits : [null]) {
        const kind = c.len === 'real' ? 'table' : 'ab';
        const spec = { ...runSpec(kind, c, 'kleinhirn'), stage: 'search', acc: false, parityFromRef: c.len !== 'real', threads,
          ...(split ? { split } : {}), keepOutputs: false };
        const r = await recorded(`${TAG}-search|${cellKey(c)}|t${threads}|${split ?? '-'}`, spec);
        rows.push({ cell: cellKey(c), browser: c.browser, short: modelOf(c.slug).short, len: c.len, threads, split,
          medianMs: r.error ? null : r.medianMs, error: r.error ?? null, parity: r.parity?.pass ?? null, runId: r.runId });
        console.log(`${cellKey(c)} t${threads} ${split ?? ''}: ${r.error ?? `${r.medianMs.toFixed(2)} ms`}`);
      }
    }
  }
  const best = {};
  for (const r of rows.filter((x) => x.medianMs !== null)) if (!best[r.cell] || r.medianMs < best[r.cell].medianMs) best[r.cell] = r;
  writeFileSync(join(RESULTS, `${TAG}-search.json`), JSON.stringify({ workorder: 'docs/R8_WORKORDER.md hc8', date: new Date().toISOString(),
    commit: ctx.commit, threadList, splits, rows, best }, null, 1));
  for (const b of Object.values(best)) console.log(`best ${b.cell}: t${b.threads} ${b.split ?? ''} ${b.medianMs.toFixed(2)} ms`);
}

// ---------------------------------------------------------------- stage: official

const OFFICIAL = resolve(ROOT, '../kleinhirn-official');
const OFFICIAL_ORDER = { even: ['kleinhirn', 'ort', 'ort', 'kleinhirn', 'kleinhirn', 'ort'],
  odd: ['ort', 'kleinhirn', 'kleinhirn', 'ort', 'ort', 'kleinhirn'] };
const RUNNER_FILES = ['src', 'bench/kbench', 'bench/kbench.ts', 'bench/kbench.html', 'bench/run-kbench.mjs',
  'bench/metrics.ts', 'data/kbench/ort-best.json', 'vite.config.ts', 'package.json'];

function officialCells(f) {
  const narrow = (all, wanted, def) => (wanted.length === all.length ? def : wanted);
  return cellMatrix({
    browsers: narrow(BROWSERS, f.browsers, ['chromium', 'safari']),
    ways: narrow(WAYS, f.ways, ['webgpu-f16', 'webgpu-f32']),
    models: f.models,
    lengths: narrow(['128', '512', 'real'], f.lengths, ['real']),
  });
}

async function stageOfficial(f) {
  const dirty = sh('git', ['-C', ROOT, 'status', '--porcelain', '--', ...RUNNER_FILES]);
  if (dirty) throw new Error(`uncommitted engine or runner files:\n${dirty}`);
  const commit = sh('git', ['-C', ROOT, 'rev-parse', '--short', 'HEAD']);
  const key = `${TAG}-official`;
  state[key] ??= { commit, cells: {} };
  const off = state[key];
  if (off.commit !== commit) throw new Error(`${key} was started at ${off.commit}, HEAD is ${commit}; finish it or pick another KBENCH_TAG`);
  if (off.originalHead) sh('git', ['-C', OFFICIAL, 'checkout', '--detach', off.originalHead]);
  if (sh('git', ['-C', OFFICIAL, 'status', '--porcelain'])) throw new Error(`${OFFICIAL} is not clean`);
  off.originalHead ??= sh('git', ['-C', OFFICIAL, 'rev-parse', 'HEAD']);
  save();
  const link = join(OFFICIAL, 'models/k28');
  if (!existsSync(link)) symlinkSync(join(ROOT, 'models/k28'), link);
  else if (!lstatSync(link).isSymbolicLink()) throw new Error(`${link} exists and is no symlink`);
  try {
    sh('git', ['-C', OFFICIAL, 'checkout', '--detach', commit]);
    execFileSync('npm', ['run', 'build'], { cwd: OFFICIAL, stdio: 'ignore' });
    serveFrom(OFFICIAL);
    ctx.commit = commit;
    killStalePid(state.vitePid);
    const v = await startVite(ctx);
    state.vitePid = v.pid;
    off.bundle = bundleBytes();
    save();
    for (const [i, c] of officialCells(f).entries()) {
      const order = OFFICIAL_ORDER[i % 2 === 0 ? 'even' : 'odd'];
      const seen = { kleinhirn: 0, ort: 0 };
      for (const engine of order) {
        const rep = seen[engine]++;
        const why = cellMissing(c, engine);
        if (why) throw new Error(`${cellKey(c)} ${engine}: ${why}`);
        const r = await recorded(`${key}|${cellKey(c)}|${engine}|${rep}`, { ...runSpec('table', c, engine), stage: 'official' });
        if (r.error) throw new Error(`${cellKey(c)} ${engine}: ${r.error}`);
        if (engine === 'kleinhirn' && r.parity?.pass === false) {
          throw new StopError(5, `kleinhirn parity fails in ${cellKey(c)}: ${r.parity.reasons.join('; ')} (${r.file})`);
        }
      }
      off.cells[cellKey(c)] = 'done';
      save();
    }
  } finally {
    stopVite(ctx);
    state.vitePid = null;
    sh('git', ['-C', OFFICIAL, 'checkout', '--detach', off.originalHead]);
    serveFrom(ROOT);
    save();
    console.log(`official worktree back on ${sh('git', ['-C', OFFICIAL, 'rev-parse', '--short', 'HEAD'])}`);
  }
  officialSummary(f);
}

function officialSummary(f) {
  const key = `${TAG}-official`;
  const off = state[key];
  const rows = [];
  for (const c of officialCells(f)) {
    const runs = (engine) => [0, 1, 2].map((rep) => state.runs[`${key}|${cellKey(c)}|${engine}|${rep}`]).filter((r) => r && !r.error);
    const side = (engine) => {
      const rs = runs(engine);
      const m = rs.map((r) => r.medianMs);
      return { runIds: rs.map((r) => r.runId), medians: m, p95s: rs.map((r) => r.p95Ms), loadStarts: rs.map((r) => r.loadStart ?? null),
        medianOfMedians: m.length ? median(m) : null, span: m.length ? [Math.min(...m), Math.max(...m)] : null };
    };
    const kh = side('kleinhirn'), ort = side('ort');
    const verdict = kh.medians.length !== 3 || ort.medians.length !== 3 ? 'incomplete'
      : Math.max(...kh.medians) < Math.min(...ort.medians) ? 'kleinhirn faster'
        : Math.max(...ort.medians) < Math.min(...kh.medians) ? 'ORT faster' : 'undecided';
    const khRuns = runs('kleinhirn');
    rows.push({ cell: cellKey(c), browser: c.browser, way: c.way, short: modelOf(c.slug).short, len: c.len,
      ortConfig: ortCfgFor(c.browser, c.way, c.slug, c.len), kleinhirn: kh, ort,
      factor: kh.medianOfMedians && ort.medianOfMedians ? ort.medianOfMedians / kh.medianOfMedians : null, verdict,
      parity: khRuns.length ? khRuns.every((r) => r.parity?.pass !== false) : null });
  }
  const doc = { workorder: 'docs/session-2026-10-06-kleinhirn-r1.md (README cells), K28.8 Festlegung 12 order',
    date: new Date().toISOString(), commit: off?.commit, officialHeadRestored: off?.originalHead, bundle: off?.bundle, rows };
  writeFileSync(join(RESULTS, `${key}-summary.json`), JSON.stringify(doc, null, 1));
  const fmt = (v) => (v == null ? '' : v.toFixed(2).replace('.', ','));
  const md = [`# ${key}: README cells, official (commit ${off?.commit})`, '',
    'Median der drei Lauf-Mediane in ms je Aufruf, Spanne in Klammern, Faktor ORT durch kleinhirn. Reihenfolge je Zelle A B B A A B.', '',
    '| Browser | Weg | Modell | kleinhirn | ORT | Faktor | Urteil | Parität | ORT-Einstellung |', '|---|---|---|---|---|---|---|---|---|',
    ...rows.map((r) => `| ${r.browser} | ${r.way.replace('webgpu-', '')} | ${r.short} | ${fmt(r.kleinhirn.medianOfMedians)} (${fmt(r.kleinhirn.span?.[0])} bis ${fmt(r.kleinhirn.span?.[1])}) | ${fmt(r.ort.medianOfMedians)} (${fmt(r.ort.span?.[0])} bis ${fmt(r.ort.span?.[1])}) | ${fmt(r.factor)} | ${r.verdict} | ${r.parity === null ? '' : r.parity ? 'ok' : 'FEHLT'} | ${r.ortConfig ? Object.entries(r.ortConfig).map(([k, v]) => `${k}=${v}`).join(' ') : ''} |`)];
  writeFileSync(join(RESULTS, `${key}-summary.md`), `${md.join('\n')}\n`);
  console.log(`official: ${rows.length} cells, ${rows.filter((r) => r.verdict === 'kleinhirn faster').length} kleinhirn faster, written ${key}-summary.json`);
}

// ---------------------------------------------------------------- main

async function main() {
  const args = process.argv.slice(2);
  const stage = args.find((a) => !a.startsWith('--')) ?? (args.includes('--all') ? 'all' : null);
  if (!stage) throw new Error('usage: run-kbench.mjs <probe|control|screen|table|mem|summary|--all> [filters]');
  const f = parseFilters(args);
  f.ortThreads = args.includes('--ort-threads') ? Number(args[args.indexOf('--ort-threads') + 1]) : null;
  const threadArg = (k) => { if (!args.includes(k)) return null; const v = args[args.indexOf(k) + 1]; return v === 'auto' ? 'auto' : Number(v); };
  f.threads = threadArg('--threads');
  f.threadsRef = threadArg('--threads-ref');
  for (let pid = process.pid; pid > 1;) {
    ctx.ownPids.add(pid);
    try { pid = Number(sh('ps', ['-o', 'ppid=', '-p', String(pid)])); } catch { break; }
  }
  ctx.commit = sh('git', ['-C', ROOT, 'rev-parse', '--short', 'HEAD']);
  const caffeinate = spawn('caffeinate', ['-dimsu', '-w', String(process.pid)], { stdio: 'ignore', detached: true });
  caffeinate.unref();
  if (stage === 'summary') { stageSummary(f); return; }
  if (stage === 'official-summary') { officialSummary(f); return; }
  if (stage === 'wasm-summary') { stageWasmSummary(f); return; }
  if (stage === 'official') { await stageOfficial(f); return; }
  killStalePid(state.vitePid);
  try {
    const v = await startVite(ctx);
    state.vitePid = v.pid;
    save();
    const order = stage === 'all' ? ['probe', 'control', 'screen', 'table', 'mem', 'summary'] : [stage];
    for (const s of order) {
      if (s === 'probe') await stageProbe(f);
      else if (s === 'control') await stageControl(f);
      else if (s === 'control-old') await stageControlOld(f);
      else if (s === 'screen') await stageScreen(f);
      else if (s === 'table') await stageTable(f);
      else if (s === 'mem') await stageMem(f);
      else if (s === 'ab') await stageAb(f);
      else if (s === 'wasm-probe') await stageWasmProbe(f);
      else if (s === 'threads-search') await stageThreadsSearch(f, args);
      else if (s === 'summary') stageSummary(f);
      else throw new Error(`unknown stage ${s}`);
    }
  } finally {
    stopVite(ctx);
    state.vitePid = null;
    save();
  }
}

try {
  await main();
} catch (e) {
  if (e instanceof PauseError) { console.error(`PAUSE: ${e.message}`); process.exit(3); }
  if (e instanceof StopError) { console.error(`STOP: ${e.message}`); process.exit(e.code); }
  throw e;
}
