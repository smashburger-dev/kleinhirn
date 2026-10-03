// K28.8 runner (docs/K28_8_WORKORDER.md, phase 1 step 9): kleinhirn
// (bench/k28-latency.html) against ONNX Runtime Web (bench/k28-ort.html), f16,
// L128 and L512, on one Vite server with cross-origin isolation
// (bench/k28/vite.k28-8.config.ts).
//
// Every run: a fresh visible Playwright Chromium, the 1-minute load below 4
// at the start (wait up to 30 minutes, then pause: state saved, exit 3), no
// foreign bench process (pgrep vite and ms-playwright, own Vite excluded;
// found: pause, exit 3), one runs.tsv line before the run, a result JSON with
// the run_id in bench/results/, the outputs of the measured calls in
// models/k28/<slug>/k28.8/outputs/<run_id>.bin. caffeinate -dimsu keeps the
// laptop awake while the runner lives.
//
// Screening (k28 worktree, change k28.8-screen, n = 100, one run per candidate):
//   --screen placement [--L 128]   node placement of every admissible graph in every
//                                  build (log=verbose, diagnostic, no latency)
//   --screen latency --L 128|512   every startable graph x build x capture, plus kleinhirn
//                                  once; at L512 the three best combinations of L128
//   --screen top3                  the three best L128 combinations per model (for the L512 export)
//   --screen tiebreak --L 128|512  candidates within 5 % of the best and the best: two more
//                                  runs each, alternating
//   --screen finite                the winner of every cell over all 320 inputs (n = 300)
//   --screen checks                cost of the K28.R checks (bench/k28-checks.html)
//   --screen best                  writes data/k28/k28.8-best.json
// Official (../kleinhirn-official at the committed k28 HEAD, change k28.8-official):
//   --official                     twelve cells, six runs each, order per Festlegung 12,
//                                  then bench/results/k28.8-official-summary.json
//   --official --dryrun            one cell (MiniLM L128), one run per side, k28.8-dryrun-*
//   --official --summary           only the summary from the state file
// State: data/k28/k28.8-state.json (atomic). Resume: run the same command again; the
// own old Vite is stopped, the official HEAD restored, an interrupted cell restarts and
// its partial runs become `superseded` in runs.tsv.

import { chromium } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  appendFileSync, existsSync, lstatSync, mkdirSync, readFileSync, renameSync, statSync,
  symlinkSync, writeFileSync,
} from 'node:fs';
import { createConnection } from 'node:net';
import { loadavg } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const K28 = resolve(fileURLToPath(new URL('..', import.meta.url)));
const OFFICIAL = resolve(K28, '../kleinhirn-official');
const CONFIG = 'bench/k28/vite.k28-8.config.ts';
const RUNS = join(K28, 'data/hillclimb/runs.tsv');
const RESULTS = join(K28, 'bench/results');
const STATE_FILE = join(K28, 'data/k28/k28.8-state.json');
const BEST_FILE = join(K28, 'data/k28/k28.8-best.json');
const MAX_LOAD = 4;
const LOAD_WAIT_MS = 30 * 60000;
const SCREEN_N = 100;
const FULL_N = 300;
const NEAR = 1.05;
const MODELS = [
  ['sentence-transformers__all-MiniLM-L6-v2', 'bert', 'embeddings'],
  ['cardiffnlp__twitter-roberta-base-sentiment-latest', 'roberta', 'sequence-classification'],
  ['cross-encoder__mmarco-mMiniLMv2-L12-H384-v1', 'xlm-roberta', 'reranking'],
  ['distilbert__distilbert-base-uncased-finetuned-sst-2-english', 'distilbert', 'sequence-classification'],
  ['protectai__deberta-v3-base-prompt-injection-v2', 'deberta-v2', 'sequence-classification'],
  ['ibm-granite__granite-embedding-small-english-r2', 'modernbert', 'embeddings'],
];
const LENGTHS = [128, 512];
const BUILDS = ['jsep', 'webgpu', 'jspi'];
// Official cells: L-major, index 0..11 (Festlegung 12 balances by index parity).
const CELLS = LENGTHS.flatMap((L) => MODELS.map(([slug]) => ({ slug, L })));
const cellKey = (slug, L) => `${slug}|L${L}`;

class PauseError extends Error {}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const load1 = () => loadavg()[0];
const sh = (cmd, a, opts = {}) => execFileSync(cmd, a, { encoding: 'utf8', ...opts }).trim();
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const taskOf = (slug) => MODELS.find(([s]) => s === slug)[2];
const familyOf = (slug) => MODELS.find(([s]) => s === slug)[1];

// ---------------------------------------------------------------- state

function readState() {
  return existsSync(STATE_FILE) ? JSON.parse(readFileSync(STATE_FILE, 'utf8')) : {};
}

function writeState(state) {
  const tmp = `${STATE_FILE}.tmp-${process.pid}`;
  writeFileSync(tmp, `${JSON.stringify(state, null, 1)}\n`);
  renameSync(tmp, STATE_FILE);
}

// ---------------------------------------------------------------- runs.tsv

const COLS = [
  'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket', 'precision',
  'argmax_agreement', 'max_abs_logit_diff', 'max_abs_prob_diff', 'median_ms',
  'p95_ms', 'model_only_median_ms', 'load_ms', 'download_mb', 'gpu_mb',
  'peak_mem_mb', 'browser', 'adapter', 'kept', 'note',
];

function appendRun(runId, fields) {
  const cells = COLS.map(() => '');
  for (const [k, v] of Object.entries({ date: new Date().toISOString().slice(0, 10), run_id: runId, kept: 'pending', ...fields })) {
    cells[COLS.indexOf(k)] = String(v).replace(/[\t\n]/g, ' ');
  }
  appendFileSync(RUNS, cells.join('\t') + '\n');
}

function finishRun(runId, fields) {
  const lines = readFileSync(RUNS, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  if (idx < 0) throw new Error(`run ${runId} missing in runs.tsv`);
  const cells = lines[idx].split('\t');
  for (const [k, v] of Object.entries(fields)) cells[COLS.indexOf(k)] = String(v).replace(/[\t\n]/g, ' ');
  lines[idx] = cells.join('\t');
  writeFileSync(RUNS, lines.join('\n'));
}

// ---------------------------------------------------------------- environment

function foreignBenchProcesses(own) {
  const out = [];
  for (const pat of ['vite', 'ms-playwright']) {
    let r = '';
    try { r = sh('pgrep', ['-fl', pat]); } catch { /* none */ }
    for (const line of r.split('\n').filter(Boolean)) {
      if (!own.has(Number(line.split(' ')[0]))) out.push(line.slice(0, 160));
    }
  }
  return out;
}

async function checkEnvironment(ctx) {
  // A closed browser can leave helpers for a moment.
  let foreign = foreignBenchProcesses(ctx.ownPids);
  for (let i = 0; i < 5 && foreign.length; i += 1) {
    await sleep(2000);
    foreign = foreignBenchProcesses(ctx.ownPids);
  }
  if (foreign.length) throw new PauseError(`foreign bench processes:\n${foreign.join('\n')}`);
  const t0 = Date.now();
  while (load1() >= MAX_LOAD) {
    if (Date.now() - t0 > LOAD_WAIT_MS) {
      throw new PauseError(`1-minute load stayed at or above ${MAX_LOAD} for 30 minutes (now ${load1().toFixed(2)})`);
    }
    await sleep(15000);
  }
  return { loadStart: Number(load1().toFixed(2)), waitedSeconds: Math.round((Date.now() - t0) / 1000) };
}

function portBusy(port) {
  return new Promise((res) => {
    const s = createConnection({ port, host: 'localhost' });
    s.on('connect', () => { s.destroy(); res(true); });
    s.on('error', () => res(false));
  });
}

async function startVite(ctx, cwd) {
  let port = 5310;
  while (await portBusy(port)) port += 1;
  const proc = spawn(join(cwd, 'node_modules/.bin/vite'), ['--config', CONFIG, '--port', String(port), '--strictPort'],
    { cwd, stdio: 'ignore', detached: true });
  ctx.ownPids.add(proc.pid);
  ctx.vite = proc;
  for (let i = 0; i < 60; i += 1) {
    if (await portBusy(port)) {
      ctx.base = `http://localhost:${port}`;
      return { pid: proc.pid, port };
    }
    await sleep(500);
  }
  throw new Error(`vite did not start on ${port}`);
}

function stopVite(ctx) {
  if (!ctx.vite) return;
  try { process.kill(-ctx.vite.pid, 'SIGTERM'); } catch { /* gone */ }
  ctx.vite = null;
}

function killStalePid(pid) {
  if (!pid) return;
  try {
    process.kill(pid, 0);
    process.kill(-pid, 'SIGTERM');
    console.log(`stopped the own old vite (pid ${pid})`);
  } catch { /* gone */ }
}

function buildIdOnDisk(root) {
  return readFileSync(join(root, 'dist/kleinhirn.js'), 'utf8').match(/buildId\s*:\s*["']([a-z0-9]+)["']/)?.[1];
}

async function checkServedBuild(ctx, root) {
  const id = buildIdOnDisk(root);
  if (!id) throw new Error('no buildId in dist/kleinhirn.js');
  const served = await (await fetch(`${ctx.base}/dist/kleinhirn.js`)).text();
  if (!served.includes(id)) throw new Error(`vite serves a stale bundle (buildId ${id})`);
  return id;
}

function bundleBytes(root) {
  const p = join(root, 'dist/kleinhirn.js');
  return { raw: statSync(p).size, gzip9: gzipSync(readFileSync(p), { level: 9 }).byteLength };
}

// ---------------------------------------------------------------- one page run

async function openPage(ctx, url, globalName, timeoutMs, onConsole) {
  const browser = await chromium.launch({ headless: false });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      if (onConsole) onConsole(m.text());
      else if (m.type() === 'error') console.log('[page]', m.text().slice(0, 300));
    });
    await page.goto(`${ctx.base}${url}`);
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      if (await page.evaluate((g) => window[g]?.done ?? false, globalName)) break;
      await sleep(500);
    }
    const result = await page.evaluate((g) => window[g], globalName);
    if (!result?.done) throw new Error(`page timeout (${result?.stage})`);
    return { result, browser: `chromium-${browser.version()}` };
  } finally {
    await browser.close();
  }
}

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

function saveOutputs(slug, runId, outputs) {
  if (!outputs) return null;
  const dir = join(K28, 'models/k28', slug, 'k28.8/outputs');
  mkdirSync(dir, { recursive: true });
  const buf = Buffer.from(outputs.base64, 'base64');
  const file = join(dir, `${runId}.bin`);
  writeFileSync(file, buf);
  return { file: file.replace(`${K28}/`, ''), sha256: sha256(buf), rows: outputs.rows, width: outputs.width,
    nonFinite: outputs.nonFinite, firstInput: outputs.firstInput };
}

// spec: { side: 'kh' | 'ort', slug, L, n, graph, build, capture }
async function measure(ctx, spec, change, prefix) {
  const env = await checkEnvironment(ctx);
  const runId = `${prefix}-${spec.side}-${spec.slug}-L${spec.L}${spec.side === 'ort'
    ? `-${spec.graph}-${spec.build}-c${spec.capture ? 1 : 0}` : ''}-${Date.now()}`;
  appendRun(runId, {
    change, commit: ctx.commit, engine: spec.side === 'kh' ? 'kleinhirn' : 'ort-web', model: spec.slug,
    bucket: `L${spec.L}`, precision: 'f16',
    note: spec.side === 'ort' ? `graph ${spec.graph} build ${spec.build} capture ${spec.capture ? 1 : 0} n ${spec.n}` : `n ${spec.n}`,
  });
  let buildId = null;
  if (spec.side === 'kh') buildId = await checkServedBuild(ctx, ctx.root);
  const url = spec.side === 'kh'
    ? `/bench/k28-latency.html?model=${spec.slug}&L=${spec.L}&n=${spec.n}`
    : `/bench/k28-ort.html?model=${spec.slug}&L=${spec.L}&graph=${spec.graph}&build=${spec.build}&capture=${spec.capture ? 1 : 0}&n=${spec.n}`;
  const global = spec.side === 'kh' ? 'khK28LatencyResult' : 'khK28OrtResult';
  let page;
  try {
    page = await openPage(ctx, url, global, 40 * 60000);
  } catch (e) {
    finishRun(runId, { kept: 'error', note: `${String(e.message ?? e).slice(0, 300)}` });
    throw e;
  }
  const { result, browser } = page;
  const outputs = saveOutputs(spec.slug, runId, result.outputs);
  delete result.outputs;
  const file = join(RESULTS, `${runId}.json`);
  const doc = {
    run_id: runId, change, commit: ctx.commit, date: new Date().toISOString(), spec, browser,
    loadStart: env.loadStart, loadWaitedSeconds: env.waitedSeconds, loadEnd: loadavg().map((v) => v.toFixed(2)).join(' '),
    buildIdDisk: buildId, outputs, ...result,
  };
  writeFileSync(file, JSON.stringify(doc, null, 1));
  if (result.error) {
    finishRun(runId, { browser, kept: 'error', note: `${file.replace(`${K28}/`, '')}; ${result.error.slice(0, 300)}` });
    return { runId, file, error: result.error, doc };
  }
  if (spec.side === 'kh' && result.info?.buildId !== buildId) {
    finishRun(runId, { kept: 'error', note: `buildId ${result.info?.buildId} != disk ${buildId}` });
    throw new Error(`page ran buildId ${result.info?.buildId}, disk ${buildId}`);
  }
  const acc = result.accuracy ?? {};
  finishRun(runId, {
    argmax_agreement: typeof (acc.argmaxAgreement ?? acc.bestPassageAgreement) === 'number'
      ? (acc.argmaxAgreement ?? acc.bestPassageAgreement).toFixed(4) : (acc.minCosine?.toFixed(6) ?? ''),
    max_abs_logit_diff: typeof (acc.maxAbsLogitDiff ?? acc.maxAbsDiff) === 'number'
      ? (acc.maxAbsLogitDiff ?? acc.maxAbsDiff).toExponential(2) : '',
    median_ms: result.latency.medianMs.toFixed(3), p95_ms: result.latency.p95Ms.toFixed(3),
    model_only_median_ms: result.latency.medianMs.toFixed(3), load_ms: Math.round(result.loadMs),
    download_mb: result.download ? (result.download.bytes / 1048576).toFixed(1) : '',
    gpu_mb: result.info?.gpuBytes ? (result.info.gpuBytes / 1048576).toFixed(1) : '',
    browser, adapter: JSON.stringify(result.info?.adapter ?? result.adapterInfo ?? {}), kept: '',
    note: `${file.replace(`${K28}/`, '')}; load ${env.loadStart}; isolated ${result.crossOriginIsolated} step ${result.timerStepMs}`,
  });
  console.log(`${runId}: median ${result.latency.medianMs.toFixed(3)} ms p95 ${result.latency.p95Ms.toFixed(3)} (load ${env.loadStart})`);
  return { runId, file, median: result.latency.medianMs, p95: result.latency.p95Ms, doc };
}

// ---------------------------------------------------------------- screening

function exportReport(slug) {
  const f = join(RESULTS, `k28.8-export-${slug}.json`);
  return existsSync(f) ? JSON.parse(readFileSync(f, 'utf8')) : null;
}

// Admissible graphs (Festlegung 9, CPU part) of a model at L with an f16 file on disk. A graph
// whose f16 file is byte-identical to an earlier one (same sha256) is the same candidate and
// runs once, under the earlier name.
function admissibleGraphs(slug, L) {
  const rep = exportReport(slug)?.lengths?.[String(L)]?.graphs ?? {};
  const seen = new Set();
  return Object.entries(rep).filter(([g, e]) => {
    if (!e.admissibleCpu || !existsSync(join(K28, 'models/k28', slug, `k28.8/onnx/L${L}-${g}-f16.onnx`))) return false;
    if (seen.has(e.f16.sha256)) return false;
    seen.add(e.f16.sha256);
    return true;
  }).map(([g]) => g);
}

const PLACED = /(All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)/;

async function placement(ctx, slug, L, graph, build) {
  const env = await checkEnvironment(ctx);
  const runId = `k28.8-placement-${slug}-L${L}-${graph}-${build}-${Date.now()}`;
  const file = join(RESULTS, `k28.8-placement-${slug}-L${L}-${graph}-${build}.json`);
  appendRun(runId, { change: 'k28.8-screen', commit: ctx.commit, engine: 'ort-web', model: slug, bucket: `L${L}`,
    precision: 'f16', note: `placement graph ${graph} build ${build}` });
  const lines = [];
  const { result, browser } = await openPage(ctx,
    `/bench/k28-ort.html?model=${slug}&L=${L}&graph=${graph}&build=${build}&capture=0&log=verbose`,
    'khK28OrtResult', 10 * 60000, (t) => lines.push(t));
  const placements = {};
  for (const l of lines) {
    const m = l.match(PLACED);
    if (m) placements[m[2]] = (placements[m[2]] ?? 0) + Number(m[3]);
  }
  const cpuNodes = [];
  let inCpu = false;
  for (const l of lines) {
    const head = l.match(/placed on \[(\w+)\]/);
    if (head) inCpu = head[1] === 'CPUExecutionProvider';
    else if (inCpu) {
      const m = l.match(/VerifyEachNodeIsAssignedToAnEp\]\s+(\w+) \((.*)\)\s*$/);
      if (m) cpuNodes.push({ name: m[2], op: m[1] }); else inCpu = false;
    }
  }
  const memcpy = lines.filter((l) => /Memcpy/.test(l));
  const doc = {
    run_id: runId, date: new Date().toISOString(), commit: ctx.commit, slug, L, graph, build, browser,
    loadStart: env.loadStart, stage: result.stage, error: result.error ?? null, validationMode: result.validationMode,
    ortVersion: result.ortVersion, crossOriginIsolated: result.crossOriginIsolated,
    placements, cpuNodes, memcpyLines: memcpy.slice(0, 50), memcpyCount: memcpy.length, consoleLines: lines.length,
    relevant: lines.filter((l) => /placed on|not assigned|fallback|capture|Memcpy|validation/i.test(l)).slice(0, 300),
  };
  writeFileSync(file, JSON.stringify(doc, null, 1));
  finishRun(runId, { browser, kept: result.error ? 'error' : 'diagnostic',
    note: `${file.replace(`${K28}/`, '')}; ${JSON.stringify(placements)}; cpu ${cpuNodes.length}; memcpy lines ${memcpy.length}${result.error ? `; ${result.error.slice(0, 200)}` : ''}` });
  console.log(`${slug} L${L} ${graph} ${build}: ${result.error ? `ERROR ${result.error.slice(0, 160)}` : JSON.stringify(placements)} cpu ${cpuNodes.length} memcpy ${memcpy.length}`);
  return { startable: !result.error, file: file.replace(`${K28}/`, ''), placements, cpuNodes: cpuNodes.length, memcpy: memcpy.length, error: result.error ?? null };
}

const comboKey = (c) => `${c.graph}|${c.build}|c${c.capture ? 1 : 0}`;

function screenCell(state, slug, L) {
  state.screen ??= { cells: {} };
  return (state.screen.cells[cellKey(slug, L)] ??= { placement: {}, ort: {}, kh: [] });
}

async function screenPlacement(ctx, state, Ls) {
  for (const L of Ls) {
    for (const [slug] of ctx.models) {
      const cell = screenCell(state, slug, L);
      const graphs = L === 128 ? admissibleGraphs(slug, L) : [...new Set((state.screen.top3?.[slug] ?? []).map((c) => c.graph))];
      for (const graph of graphs) {
        for (const build of BUILDS) {
          const k = `${graph}|${build}`;
          if (cell.placement[k]) continue;
          cell.placement[k] = await placement(ctx, slug, L, graph, build);
          writeState(state);
        }
      }
    }
  }
}

function recordOrt(cell, combo, r) {
  const e = (cell.ort[comboKey(combo)] ??= { ...combo, runs: [] });
  e.runs.push(r.error ? { runId: r.runId, file: r.file.replace(`${K28}/`, ''), error: r.error }
    : { runId: r.runId, file: r.file.replace(`${K28}/`, ''), median: r.median, p95: r.p95, accuracy: r.doc.accuracy,
      nonFinite: r.doc.outputs?.nonFinite, loadStart: r.doc.loadStart });
}

const runMedians = (e) => e.runs.filter((r) => r.median !== undefined).map((r) => r.median);
const comboScore = (e) => (runMedians(e).length ? median(runMedians(e)) : Infinity);

async function screenLatency(ctx, state, L) {
  for (const [slug] of ctx.models) {
    const cell = screenCell(state, slug, L);
    let combos;
    if (L === 128) {
      combos = Object.values(cell.placement).length ? Object.entries(cell.placement)
        .filter(([, p]) => p.startable)
        .flatMap(([k]) => { const [graph, build] = k.split('|'); return [0, 1].map((c) => ({ graph, build, capture: c === 1 })); })
        : [];
    } else {
      combos = (state.screen.top3?.[slug] ?? []).map(({ graph, build, capture }) => ({ graph, build, capture }));
      for (const c of combos) {
        if (!existsSync(join(K28, 'models/k28', slug, `k28.8/onnx/L512-${c.graph}-f16.onnx`))) {
          throw new Error(`${slug}: L512 graph ${c.graph} missing; run convert/k28_8_export.py ${slug} 512 ${c.graph}`);
        }
      }
    }
    if (!combos.length) throw new Error(`${slug} L${L}: no candidates (placement first?)`);
    for (const combo of combos) {
      if (cell.ort[comboKey(combo)]?.runs.length) continue;
      const r = await measure(ctx, { side: 'ort', slug, L, n: SCREEN_N, ...combo }, 'k28.8-screen', 'k28.8-screen');
      recordOrt(cell, combo, r);
      writeState(state);
    }
    if (!cell.kh.length) {
      const r = await measure(ctx, { side: 'kh', slug, L, n: SCREEN_N }, 'k28.8-screen', 'k28.8-screen');
      cell.kh.push({ runId: r.runId, file: r.file.replace(`${K28}/`, ''), median: r.median, p95: r.p95,
        gpuBytes: r.doc.info?.gpuBytes, accuracy: r.doc.accuracy, error: r.error });
      writeState(state);
    }
  }
}

function top3(state) {
  state.screen.top3 = {};
  for (const [slug] of MODELS) {
    const cell = state.screen.cells[cellKey(slug, 128)];
    if (!cell) continue;
    const ranked = Object.values(cell.ort).filter((e) => Number.isFinite(comboScore(e)))
      .sort((a, b) => comboScore(a) - comboScore(b));
    state.screen.top3[slug] = ranked.slice(0, 3).map((e) => ({ graph: e.graph, build: e.build, capture: e.capture, l128Median: comboScore(e) }));
    console.log(`${slug}: ${state.screen.top3[slug].map((c) => `${c.graph}/${c.build}/c${c.capture ? 1 : 0} ${c.l128Median.toFixed(3)}`).join(', ')}`);
  }
}

async function screenTiebreak(ctx, state, L) {
  for (const [slug] of ctx.models) {
    const cell = screenCell(state, slug, L);
    const ranked = Object.values(cell.ort).filter((e) => Number.isFinite(comboScore(e)))
      .sort((a, b) => comboScore(a) - comboScore(b));
    if (!ranked.length) continue;
    // Decided on the first screening run of each candidate; resumable per candidate.
    cell.tiebreak ??= (() => {
      const best = ranked[0];
      const firstBest = runMedians(best)[0];
      return { best: comboKey(best), near: ranked.slice(1).filter((e) => runMedians(e)[0] <= firstBest * NEAR).map(comboKey), done: [] };
    })();
    const best = cell.ort[cell.tiebreak.best];
    for (const k of cell.tiebreak.near) {
      if (cell.tiebreak.done.includes(k)) continue;
      for (const e of [best, cell.ort[k], best, cell.ort[k]]) {
        const r = await measure(ctx, { side: 'ort', slug, L, n: SCREEN_N, graph: e.graph, build: e.build, capture: e.capture },
          'k28.8-screen', 'k28.8-screen');
        recordOrt(cell, e, r);
        writeState(state);
      }
      cell.tiebreak.done.push(k);
    }
    const near = cell.tiebreak.near.map((k) => cell.ort[k]);
    writeState(state);
    console.log(`${slug} L${L}: best ${comboKey(best)}, near ${near.map(comboKey).join(', ') || 'none'}`);
  }
}

function winnerOf(cell) {
  const ranked = Object.values(cell.ort).filter((e) => Number.isFinite(comboScore(e)))
    .sort((a, b) => comboScore(a) - comboScore(b));
  return ranked[0] ?? null;
}

async function screenFinite(ctx, state) {
  for (const L of LENGTHS) {
    for (const [slug] of ctx.models) {
      const cell = screenCell(state, slug, L);
      const w = winnerOf(cell);
      if (!w) throw new Error(`${slug} L${L}: no winner`);
      if (cell.finite?.combo === comboKey(w)) continue;
      const r = await measure(ctx, { side: 'ort', slug, L, n: FULL_N, graph: w.graph, build: w.build, capture: w.capture },
        'k28.8-screen', 'k28.8-finite');
      cell.finite = { combo: comboKey(w), runId: r.runId, file: r.file.replace(`${K28}/`, ''), error: r.error ?? null,
        warmupFinite: r.doc.warmupFinite, nonFinite: r.doc.outputs?.nonFinite, rows: r.doc.outputs?.rows,
        accuracy: r.doc.accuracy, median: r.median, download: r.doc.download };
      writeState(state);
    }
  }
}

async function screenChecks(ctx, state) {
  const env = await checkEnvironment(ctx);
  const runId = `k28.8-checks-${Date.now()}`;
  appendRun(runId, { change: 'k28.8-screen', commit: ctx.commit, engine: 'kleinhirn', model: 'none', bucket: '', precision: '',
    note: 'cost of validation scope and assertFinite' });
  const { result, browser } = await openPage(ctx, '/bench/k28-checks.html', 'khK28ChecksResult', 10 * 60000);
  const file = join(RESULTS, 'k28.8-checks.json');
  writeFileSync(file, JSON.stringify({ run_id: runId, commit: ctx.commit, date: new Date().toISOString(), browser,
    loadStart: env.loadStart, ...result }, null, 1));
  finishRun(runId, { browser, kept: result.error ? 'error' : 'diagnostic', note: `${file.replace(`${K28}/`, '')}; ${JSON.stringify(result.scope ?? result.error)}` });
  state.screen.checks = { file: file.replace(`${K28}/`, ''), scope: result.scope, finite: result.finite, error: result.error ?? null };
  writeState(state);
  console.log(JSON.stringify(state.screen.checks));
}

// K27: kleinhirn only, once per model at L (n = 100), against the ORT screening winners of
// k28.8-best.json. Results under state.k27[commit]; change k27-kh.
async function screenKh(ctx, state, L) {
  const best = JSON.parse(readFileSync(BEST_FILE, 'utf8'));
  state.k27 ??= {};
  const mine = (state.k27[ctx.commit] ??= {});
  for (const [slug] of ctx.models) {
    const k = cellKey(slug, L);
    if (mine[k]) continue;
    const r = await measure(ctx, { side: 'kh', slug, L, n: SCREEN_N }, 'k27-kh', 'k27-kh');
    const ort = best.cells[k].winner.screeningMedian;
    mine[k] = { runId: r.runId, median: r.median, p95: r.p95, ort, ratio: r.median / ort, accuracy: r.doc.accuracy, error: r.error ?? null };
    writeState(state);
    console.log(`${k}: kleinhirn ${r.median?.toFixed(3)} ms, ORT ${ort.toFixed(3)} ms, ratio ${(r.median / ort).toFixed(3)}`);
  }
}

// K27: kleinhirn and the K28.8 ORT winner back to back per cell (A B, odd cells B A), n = 100,
// so both sides run under the same conditions. Not official (one run per side). change k27-duel.
async function screenDuel(ctx, state, L) {
  const best = JSON.parse(readFileSync(BEST_FILE, 'utf8'));
  state.k27duel ??= {};
  const mine = (state.k27duel[ctx.commit] ??= {});
  for (const [idx, [slug]] of ctx.models.entries()) {
    const k = cellKey(slug, L);
    if (mine[k]?.ort && mine[k]?.kh) continue;
    const w = best.cells[k].winner;
    const order = idx % 2 === 0 ? ['kh', 'ort'] : ['ort', 'kh'];
    mine[k] = { order, ortConfig: w };
    for (const side of order) {
      const spec = side === 'kh' ? { side: 'kh', slug, L, n: SCREEN_N }
        : { side: 'ort', slug, L, n: SCREEN_N, graph: w.graph, build: w.build, capture: w.capture };
      const r = await measure(ctx, spec, 'k27-duel', 'k27-duel');
      mine[k][side] = { runId: r.runId, median: r.median, p95: r.p95, accuracy: r.doc.accuracy, loadStart: r.doc.loadStart, error: r.error ?? null };
      writeState(state);
    }
    console.log(`${k}: kleinhirn ${mine[k].kh.median?.toFixed(3)} ms, ORT ${mine[k].ort.median?.toFixed(3)} ms, ratio ${(mine[k].kh.median / mine[k].ort.median).toFixed(3)}`);
  }
}

function writeBest(state) {
  const cells = {};
  let ortCalls = 0; let khCalls = 0;
  for (const { slug, L } of CELLS) {
    const cell = state.screen.cells[cellKey(slug, L)];
    const w = winnerOf(cell);
    const exp = exportReport(slug)?.lengths?.[String(L)]?.graphs?.[w.graph];
    const kh = cell.kh.find((r) => r.median !== undefined);
    cells[cellKey(slug, L)] = {
      slug, L, family: familyOf(slug), task: taskOf(slug),
      winner: { graph: w.graph, build: w.build, capture: w.capture, validationMode: w.build === 'jsep' ? null : 'wgpuOnly',
        screeningMedian: comboScore(w), runs: w.runs.length },
      candidates: Object.values(cell.ort).map((e) => ({ graph: e.graph, build: e.build, capture: e.capture,
        medians: runMedians(e), score: Number.isFinite(comboScore(e)) ? comboScore(e) : null,
        error: e.runs.find((r) => r.error)?.error?.slice(0, 200) ?? null })),
      placement: cell.placement[`${w.graph}|${w.build}`] ?? state.screen.cells[cellKey(slug, 128)].placement[`${w.graph}|${w.build}`] ?? null,
      cpuParity: exp ? { f32: exp.f32.parityCpu, f16: exp.f16.parityCpu, nodes: exp.f16.nodes, fused: exp.f16.fused, bytes: exp.f16.bytes } : null,
      browserAccuracy: cell.finite?.accuracy ?? null,
      finite: cell.finite ? { combo: cell.finite.combo, runId: cell.finite.runId, warmupFinite: cell.finite.warmupFinite,
        nonFinite: cell.finite.nonFinite, rows: cell.finite.rows, admissible: cell.finite.combo === comboKey(w)
          && !cell.finite.error && cell.finite.warmupFinite && cell.finite.nonFinite === 0 } : null,
      tiebreak: cell.tiebreak ?? null,
      kleinhirnScreening: kh ? { median: kh.median, p95: kh.p95, gpuBytes: kh.gpuBytes, accuracy: kh.accuracy } : null,
    };
    ortCalls += comboScore(w) * 3 * 320; // three ORT runs per cell, 320 calls each
    khCalls += (kh?.median ?? 0) * 3 * 320;
  }
  const computeMin = (ortCalls + khCalls) / 60000;
  const doc = {
    workorder: 'docs/K28_8_WORKORDER.md, phase 1 step 10', date: new Date().toISOString(), commit: state.screen.commit,
    rule: 'winner: smallest median of its screening run medians (n = 100); candidates within 5 % of the best ran twice more, alternating',
    cells,
    checks: state.screen.checks ?? null,
    notRun: CELLS.some(({ slug, L }) => !state.screen.cells[cellKey(slug, L)].finite)
      ? 'stage 3 (tiebreak), the finiteness run over 320 inputs and the dry run were not run: Noa dropped '
        + 'phase 2 on 02.10. after the screening (kleinhirn slower in all twelve cells); winners are the '
        + 'smallest single screening median' : null,
    phase2Estimate: {
      runs: 72, computeMinutes: Number(computeMin.toFixed(1)),
      note: 'compute = 320 calls x screening median per run; browser starts, loading and load waits come on top',
    },
  };
  writeFileSync(BEST_FILE, `${JSON.stringify(doc, null, 1)}\n`);
  console.log(`wrote ${BEST_FILE.replace(`${K28}/`, '')}; compute estimate ${computeMin.toFixed(1)} min`);
}

// ---------------------------------------------------------------- official

const ORDER = { even: ['A', 'B', 'B', 'A', 'A', 'B'], odd: ['B', 'A', 'A', 'B', 'B', 'A'] };

function supersede(runIds) {
  for (const id of runIds) {
    try { finishRun(id, { kept: 'superseded' }); } catch { /* not written */ }
  }
}

function ensureOfficialModels() {
  const link = join(OFFICIAL, 'models/k28');
  if (!existsSync(link)) {
    symlinkSync(join(K28, 'models/k28'), link);
    console.log(`symlink ${link} -> ${join(K28, 'models/k28')}`);
  } else if (!lstatSync(link).isSymbolicLink()) {
    throw new Error(`${link} exists and is no symlink`);
  }
}

function compareOutputs(slug, L, outputs) {
  if (!outputs) return null;
  const meta = JSON.parse(readFileSync(join(K28, 'models/k28', slug, `k28.8/inputs-L${L}.json`), 'utf8'));
  const refBuf = readFileSync(join(K28, 'models/k28', slug, `k28.8/cpu-f32-L${L}.bin`));
  const ref = new Float32Array(refBuf.buffer, refBuf.byteOffset, refBuf.byteLength / 4);
  const gotBuf = readFileSync(join(K28, outputs.file));
  const got = new Float32Array(gotBuf.buffer, gotBuf.byteOffset, gotBuf.byteLength / 4);
  const w = outputs.width;
  if (ref.length !== meta.rows * w) throw new Error(`${slug} L${L}: reference width`);
  let maxAbs = 0; let argmaxSame = 0; let minCos = 1;
  for (let r = 0; r < outputs.rows; r += 1) {
    const a = got.subarray(r * w, (r + 1) * w);
    const b = ref.subarray((outputs.firstInput + r) * w, (outputs.firstInput + r + 1) * w);
    let dot = 0; let na = 0; let nb = 0; let ia = 0; let ib = 0;
    for (let k = 0; k < w; k += 1) {
      maxAbs = Math.max(maxAbs, Math.abs(a[k] - b[k]));
      dot += a[k] * b[k]; na += a[k] * a[k]; nb += b[k] * b[k];
      if (a[k] > a[ia]) ia = k;
      if (b[k] > b[ib]) ib = k;
    }
    if (ia === ib) argmaxSame += 1;
    minCos = Math.min(minCos, dot / Math.sqrt(na * nb));
  }
  const task = taskOf(slug);
  return { rows: outputs.rows, maxAbsDiff: maxAbs,
    ...(task === 'sequence-classification' ? { argmaxAgreement: argmaxSame / outputs.rows } : {}),
    ...(task === 'embeddings' ? { minCosine: minCos } : {}) };
}

function verdict(aMedians, bMedians) {
  if (aMedians.length !== 3 || bMedians.length !== 3) return 'incomplete';
  if (Math.max(...aMedians) < Math.min(...bMedians)) return 'kleinhirn faster';
  if (Math.max(...bMedians) < Math.min(...aMedians)) return 'ORT faster';
  return 'undecided';
}

function officialSummary(state, key, outName) {
  const off = state[key];
  const cells = {};
  for (const [i, { slug, L }] of CELLS.entries()) {
    const c = off?.cells?.[cellKey(slug, L)];
    if (!c) continue;
    const docs = c.runs.map((r) => JSON.parse(readFileSync(join(K28, r.file), 'utf8')));
    const side = (s) => docs.filter((d) => d.spec.side === (s === 'A' ? 'kh' : 'ort') && !d.error);
    const per = (d) => ({ runId: d.run_id, median: d.latency.medianMs, p95: d.latency.p95Ms, loadStart: d.loadStart,
      timerStepMs: d.timerStepMs, crossOriginIsolated: d.crossOriginIsolated, vsCpuF32: compareOutputs(slug, L, d.outputs),
      accuracy: d.accuracy });
    const A = side('A').map(per); const B = side('B').map(per);
    const am = A.map((x) => x.median); const bm = B.map((x) => x.median);
    const kh = side('A')[0]; const ort = side('B')[0];
    cells[cellKey(slug, L)] = {
      index: i, slug, L, order: c.order, ortConfig: c.ortConfig,
      kleinhirn: { runs: A, medianOfMedians: am.length ? median(am) : null, span: am.length ? [Math.min(...am), Math.max(...am)] : null,
        gpuBytes: kh?.info?.gpuBytes ?? null, download: kh?.download ?? null },
      ort: { runs: B, medianOfMedians: bm.length ? median(bm) : null, span: bm.length ? [Math.min(...bm), Math.max(...bm)] : null,
        download: ort?.download ?? null },
      factor: am.length && bm.length ? median(bm) / median(am) : null,
      verdict: verdict(am, bm),
    };
  }
  const all = Object.values(cells);
  const doc = {
    workorder: 'docs/K28_8_WORKORDER.md, Festlegung 11', date: new Date().toISOString(), commit: off?.commit,
    officialHeadRestored: off?.originalHead, cells,
    gate2: all.length === 12 && all.every((c) => c.verdict === 'kleinhirn faster'),
    verdicts: Object.fromEntries(all.map((c) => [cellKey(c.slug, c.L), c.verdict])),
  };
  writeFileSync(join(RESULTS, outName), `${JSON.stringify(doc, null, 1)}\n`);
  console.log(`wrote bench/results/${outName}`);
  for (const c of all) console.log(`${c.slug} L${c.L}: ${c.verdict} factor ${c.factor?.toFixed(3)}`);
}

async function official(ctx, state, dryrun) {
  const key = dryrun ? 'dryrun' : 'official';
  const change = dryrun ? 'k28.8-dryrun' : 'k28.8-official';
  const prefix = dryrun ? 'k28.8-dryrun' : 'k28.8-official';
  const best = JSON.parse(readFileSync(BEST_FILE, 'utf8'));
  const dirtyK28 = sh('git', ['-C', K28, 'status', '--porcelain', '--', 'src', 'bench/k28', 'bench/k28-ort.ts',
    'bench/k28-latency.ts', 'bench/k28-ort.html', 'bench/k28-latency.html', 'bench/run-k28-8.mjs', 'bench/metrics.ts',
    'data/k28/k28.8-best.json', 'vite.config.ts', 'package.json']);
  if (dirtyK28) throw new Error(`k28 worktree has uncommitted runner files:\n${dirtyK28}`);
  const commit = sh('git', ['-C', K28, 'rev-parse', '--short', 'HEAD']);
  if (dryrun && state.dryrun) {
    killStalePid(state.dryrun.vitePid);
    if (state.dryrun.originalHead) sh('git', ['-C', OFFICIAL, 'checkout', '--detach', state.dryrun.originalHead]);
    delete state.dryrun; // a smoke test starts over
  }
  state[key] ??= { commit, cells: {} };
  const off = state[key];
  killStalePid(off.vitePid);
  if (off.originalHead) {
    sh('git', ['-C', OFFICIAL, 'checkout', '--detach', off.originalHead]);
  }
  const dirty = sh('git', ['-C', OFFICIAL, 'status', '--porcelain']);
  if (dirty) throw new Error(`official worktree is not clean:\n${dirty}`);
  off.originalHead ??= sh('git', ['-C', OFFICIAL, 'rev-parse', 'HEAD']);
  if (off.commit !== commit) throw new Error(`state was started at ${off.commit}, k28 HEAD is ${commit}; finish or move the state file`);
  writeState(state);
  ctx.commit = commit;
  ctx.root = OFFICIAL;
  try {
    sh('git', ['-C', OFFICIAL, 'checkout', '--detach', commit]);
    execFileSync('npm', ['run', 'build'], { cwd: OFFICIAL, stdio: 'ignore' });
    ensureOfficialModels();
    const v = await startVite(ctx, OFFICIAL);
    off.vitePid = v.pid;
    off.bundle = bundleBytes(OFFICIAL);
    writeState(state);
    const cells = dryrun ? CELLS.slice(0, 1) : CELLS;
    for (const [i, { slug, L }] of cells.entries()) {
      const k = cellKey(slug, L);
      const c = off.cells[k];
      if (c?.status === 'done') continue;
      if (c?.status === 'running') {
        supersede(c.runs.map((r) => r.runId));
        console.log(`${k}: restarting, ${c.runs.length} partial runs superseded`);
      }
      const w = best.cells[k].winner;
      const order = dryrun ? ['A', 'B'] : ORDER[i % 2 === 0 ? 'even' : 'odd'];
      off.cells[k] = { status: 'running', order, ortConfig: w, runs: [] };
      writeState(state);
      for (const s of order) {
        const spec = s === 'A' ? { side: 'kh', slug, L, n: FULL_N }
          : { side: 'ort', slug, L, n: FULL_N, graph: w.graph, build: w.build, capture: w.capture };
        const r = await measure(ctx, spec, change, prefix);
        if (r.error) throw new Error(`${r.runId}: ${r.error}`);
        off.cells[k].runs.push({ side: s, runId: r.runId, file: r.file.replace(`${K28}/`, '') });
        writeState(state);
      }
      off.cells[k].status = 'done';
      writeState(state);
    }
  } finally {
    stopVite(ctx);
    sh('git', ['-C', OFFICIAL, 'checkout', '--detach', off.originalHead]);
    off.vitePid = null;
    writeState(state);
    console.log(`official worktree back on ${sh('git', ['-C', OFFICIAL, 'rev-parse', '--short', 'HEAD'])}`);
  }
  officialSummary(state, key, dryrun ? 'k28.8-dryrun-summary.json' : 'k28.8-official-summary.json');
}

// ---------------------------------------------------------------- main

async function main() {
  const args = process.argv.slice(2);
  const val = (k) => (args.includes(k) ? args[args.indexOf(k) + 1] : null);
  const state = readState();
  // K28_8_OWN_PIDS: extra own processes (a Vite of this session) the process check ignores.
  const ctx = { ownPids: new Set((process.env.K28_8_OWN_PIDS ?? '').split(',').filter(Boolean).map(Number)),
    vite: null, base: null, commit: null, root: K28 };
  // The runner's own ancestors (the shell that started it may carry 'vite' in its command line).
  for (let pid = process.pid; pid > 1;) {
    ctx.ownPids.add(pid);
    try { pid = Number(sh('ps', ['-o', 'ppid=', '-p', String(pid)])); } catch { break; }
  }
  const only = val('--models');
  ctx.models = only ? MODELS.filter(([s]) => only.split(',').includes(s)) : MODELS;
  const caffeinate = spawn('caffeinate', ['-dimsu', '-w', String(process.pid)], { stdio: 'ignore', detached: true });
  caffeinate.unref();
  if (args.includes('--official')) {
    if (args.includes('--summary')) { officialSummary(state, 'official', 'k28.8-official-summary.json'); return; }
    await official(ctx, state, args.includes('--dryrun'));
    return;
  }
  const stage = val('--screen');
  if (!stage) throw new Error('usage: --screen <stage> | --official [--dryrun|--summary]');
  state.screen ??= { cells: {} };
  if (stage === 'top3') { top3(state); writeState(state); return; }
  if (stage === 'best') { writeBest(state); return; }
  const dirty = sh('git', ['-C', K28, 'status', '--porcelain', '--', 'src', 'bench/k28', 'bench/k28-ort.ts', 'bench/k28-latency.ts']);
  if (dirty && !args.includes('--allow-dirty')) throw new Error(`commit the pages and src first:\n${dirty}`);
  ctx.commit = sh('git', ['-C', K28, 'rev-parse', '--short', 'HEAD']);
  state.screen.commit = ctx.commit;
  killStalePid(state.screen.vitePid);
  try {
    const v = await startVite(ctx, K28);
    state.screen.vitePid = v.pid;
    writeState(state);
    const L = Number(val('--L') ?? '128');
    if (stage === 'placement') await screenPlacement(ctx, state, [L]);
    else if (stage === 'latency') await screenLatency(ctx, state, L);
    else if (stage === 'tiebreak') await screenTiebreak(ctx, state, L);
    else if (stage === 'finite') await screenFinite(ctx, state);
    else if (stage === 'checks') await screenChecks(ctx, state);
    else if (stage === 'kh') await screenKh(ctx, state, L);
    else if (stage === 'duel') await screenDuel(ctx, state, L);
    else throw new Error(`unknown stage ${stage}`);
  } finally {
    stopVite(ctx);
    state.screen.vitePid = null;
    writeState(state);
  }
}

try {
  await main();
} catch (e) {
  if (e instanceof PauseError) {
    console.error(`PAUSE: ${e.message}`);
    process.exit(3);
  }
  throw e;
}
