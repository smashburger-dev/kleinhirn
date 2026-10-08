// R1 runner library (docs/R1_WORKORDER.md): constants, the cell matrix, the bucket rule, the
// table summary, and the process helpers shared by bench/run-kbench.mjs (load rule, foreign
// process check, Vite with isolation, runs.tsv, browser drivers). The pure parts have unit
// tests (tests/kbench.test.mjs). bench/run-k28-8.mjs stays untouched; its helpers are copied.

import { chromium, firefox, webkit } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import {
  appendFileSync, existsSync, mkdtempSync, readFileSync, renameSync, rmSync, statSync, writeFileSync,
} from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { createConnection, createServer } from 'node:net';
import { loadavg, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

export const ROOT = resolve(fileURLToPath(new URL('../..', import.meta.url)));
export const RUNS = join(ROOT, 'data/hillclimb/runs.tsv');
export const RESULTS = join(ROOT, 'bench/results');
export const STATE_FILE = join(ROOT, 'data/kbench/state.json');
// The worktree Vite serves and builds from: ROOT, or ../kleinhirn-official for the stage official
// (committed code only). Results, state and runs.tsv stay under ROOT.
let serveRoot = ROOT;
export const serveFrom = (dir) => { serveRoot = dir; };
export const servedRoot = () => serveRoot;
export const VITE_CONFIG = 'bench/kbench/vite.kbench.config.ts';
// KBENCH_MAX_LOAD: only for the memory pass, whose footprint does not depend on CPU load.
export const MAX_LOAD = Number(process.env.KBENCH_MAX_LOAD ?? 4);
export const LOAD_WAIT_MS = 30 * 60000;

export const MODELS = [
  { slug: 'sentence-transformers__all-MiniLM-L6-v2', short: 'MiniLM', family: 'bert', task: 'embeddings', normalized: true },
  { slug: 'cardiffnlp__twitter-roberta-base-sentiment-latest', short: 'RoBERTa-base', family: 'roberta', task: 'sequence-classification' },
  { slug: 'cross-encoder__mmarco-mMiniLMv2-L12-H384-v1', short: 'mMiniLM', family: 'xlm-roberta', task: 'reranking' },
  { slug: 'distilbert__distilbert-base-uncased-finetuned-sst-2-english', short: 'DistilBERT', family: 'distilbert', task: 'sequence-classification' },
  { slug: 'protectai__deberta-v3-base-prompt-injection-v2', short: 'DeBERTa-v3-base', family: 'deberta-v2', task: 'sequence-classification' },
  { slug: 'ibm-granite__granite-embedding-small-english-r2', short: 'granite', family: 'modernbert', task: 'embeddings' },
];
export const BROWSERS = ['chromium', 'webkit', 'firefox', 'safari'];
// firefox-reg (R8 Festlegung 9): a regular Firefox without a debugger connection, WASM cells only,
// only when a filter names it (the R1 and R2 matrices stay as they were).
export const EXTRA_BROWSERS = ['firefox-reg'];
export const WAYS = ['webgpu-f16', 'webgpu-f32', 'wasm'];
export const LENGTHS = ['128', '512', 'real'];
export const BUCKETS = [128, 512];
export const ENGINES = ['kleinhirn', 'ort'];

export const modelOf = (slug) => MODELS.find((m) => m.slug === slug);

// ---------------------------------------------------------------- pure helpers

// Smallest bucket that holds `length` tokens (Festlegung 1; same rule in bench/kbench/engine.ts).
export function bucketFor(length, buckets = BUCKETS) {
  const hit = [...buckets].sort((a, b) => a - b).find((b) => length <= b);
  if (hit === undefined) throw new Error(`length ${length} exceeds the largest bucket`);
  return hit;
}

// Block size from the timer step (Festlegung 5).
export const blockForTimerStep = (stepMs) => (stepMs > 0.1 ? 10 : 1);

// Filter values from the command line: comma lists; a missing filter keeps everything.
export function pick(all, value, label) {
  if (!value) return all;
  const want = value.split(',');
  for (const w of want) if (!all.includes(w)) throw new Error(`${label}: ${w} is none of ${all.join(', ')}`);
  return all.filter((a) => want.includes(a));
}

export function parseFilters(args) {
  const val = (k) => (args.includes(k) ? args[args.indexOf(k) + 1] : null);
  const models = val('--models');
  const slugs = MODELS.map((m) => m.slug);
  const shorts = Object.fromEntries(MODELS.map((m) => [m.short.toLowerCase(), m.slug]));
  const wanted = models ? models.split(',').map((m) => shorts[m.toLowerCase()] ?? m) : null;
  return {
    browsers: val('--browsers') ? pick([...BROWSERS, ...EXTRA_BROWSERS], val('--browsers'), 'browsers') : BROWSERS,
    ways: pick(WAYS, val('--ways')?.split(',').map((w) => (w === 'wasm' ? w : `webgpu-${w.replace('webgpu-', '')}`)).join(','), 'ways'),
    models: wanted ? pick(slugs, wanted.join(','), 'models') : slugs,
    lengths: pick(LENGTHS, val('--lengths'), 'lengths'),
    engines: pick(ENGINES, val('--engines'), 'engines'),
  };
}

// Table cells in run order: browser, way, then model and length (Phase 2).
export function cellMatrix({ browsers, ways, models, lengths }) {
  const cells = [];
  for (const browser of [...BROWSERS, ...EXTRA_BROWSERS].filter((b) => browsers.includes(b))) {
    for (const way of WAYS.filter((w) => ways.includes(w) && (!EXTRA_BROWSERS.includes(browser) || w === 'wasm'))) {
      for (const slug of models) for (const len of LENGTHS.filter((l) => lengths.includes(l))) {
        cells.push({ browser, way, slug, len });
      }
    }
  }
  return cells;
}

// Engine order within a cell alternates with the cell index (Festlegung 4).
export const engineOrder = (index) => (index % 2 === 0 ? ['kleinhirn', 'ort'] : ['ort', 'kleinhirn']);

export const cellKey = (c) => `${c.browser}|${c.way}|${c.slug}|${c.len}`;

export function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Why an engine has no run in a cell, or null when it can run. `probe` is the entry of
// bench/results/r1-probe.json for the browser (null when not probed).
export function missingReason(browser, way, engine, probe) {
  if (!probe) return 'fehlt: Browser nicht geprobt';
  if (probe.automationError) return `fehlt: Automatisierung (${probe.automationError})`;
  if (way !== 'wasm') {
    if (!probe.env?.webgpu) return 'fehlt: kein WebGPU';
    if (way === 'webgpu-f16' && !probe.env?.adapter?.shaderF16) return 'fehlt: kein shader-f16';
  }
  const cap = probe.capability?.[engine === 'kleinhirn' ? 'kleinhirn' : 'ort']?.[way];
  if (cap && cap.startable === false) return `fehlt: ${engine === 'kleinhirn' ? 'kleinhirn' : 'ORT'} startet nicht (${String(cap.error ?? '').slice(0, 160)})`;
  return null;
}

export function gzipBytes(file) {
  return existsSync(file) ? gzipSync(readFileSync(file), { level: 9 }).byteLength : null;
}

// gzip -9 of the engine files a run loaded (paths as the page reports them).
export function engineGzip(root, engine, files) {
  if (engine === 'kleinhirn') return { bytes: gzipBytes(join(root, 'dist/kleinhirn.js')), files: ['dist/kleinhirn.js'] };
  if (engine === 'kleinhirn-ref') return { bytes: gzipBytes(join(root, 'dist-ref/kleinhirn.js')), files: ['dist-ref/kleinhirn.js'] };
  const used = [];
  let bytes = 0;
  for (const f of files ?? []) {
    const p = join(root, f.replace(/^\//, '').split('?')[0]);
    const g = gzipBytes(p);
    if (g !== null) { bytes += g; used.push(f); }
  }
  return { bytes: used.length ? bytes : null, files: used };
}

// Output rows (base64 float32) against the K28.8 CPU f32 reference of the full-length inputs
// (models/k28/<slug>/k28.8/cpu-f32-L<L>.bin, rows indexed by input number).
export function compareToReference(outputs, refBuf, firstInput, width, task) {
  const raw = Buffer.from(outputs.base64, 'base64'); // pooled buffer: respect byteOffset
  const got = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const ref = new Float32Array(refBuf.buffer.slice(refBuf.byteOffset, refBuf.byteOffset + refBuf.byteLength));
  const w = outputs.width;
  if (w !== width) throw new Error(`output width ${w} != reference width ${width}`);
  let maxAbs = 0; let same = 0; let minCos = 1;
  for (let r = 0; r < outputs.rows; r += 1) {
    const a = got.subarray(r * w, (r + 1) * w);
    const b = ref.subarray((firstInput + r) * w, (firstInput + r + 1) * w);
    let dot = 0; let na = 0; let nb = 0; let ia = 0; let ib = 0;
    for (let k = 0; k < w; k += 1) {
      maxAbs = Math.max(maxAbs, Math.abs(a[k] - b[k]));
      dot += a[k] * b[k]; na += a[k] * a[k]; nb += b[k] * b[k];
      if (a[k] > a[ia]) ia = k;
      if (b[k] > b[ib]) ib = k;
    }
    if (ia === ib) same += 1;
    minCos = Math.min(minCos, dot / Math.sqrt(na * nb));
  }
  return { rows: outputs.rows, maxAbsDiff: maxAbs,
    ...(task === 'sequence-classification' ? { argmaxAgreement: same / outputs.rows } : {}),
    ...(task === 'embeddings' ? { minCosine: minCos } : {}) };
}

// Parity rule for kleinhirn (Festlegung 6). f32: the K28 gate (argmax and best passage 100 %,
// embedding cosine >= 0.9999). f16: the largest deviation within 3 x delta_sim of the f16
// simulation (FINDINGS §42, bench/results/k28-f16-sim/<slug>.json), decisions at least 99.5 %
// (the "not close" rate of K28.S3), embedding cosine >= 0.9999. `acc` is the page's golden
// accuracy, `vsRef` the optional comparison against the CPU f32 reference. The simulation delta of
// an embedding model with an L2-normalizing head (MiniLM) is measured on the normalized vector, the
// page compares the vector before the norm, so there the absolute bound does not apply and the
// cosine decides.
export function parityVerdict(way, task, acc, vsRef, deltaSim, normalized = false) {
  const reasons = [];
  const f16 = way === 'webgpu-f16';
  const decision = acc.argmaxAgreement ?? acc.bestPassageAgreement;
  const need = f16 ? 0.995 : 1;
  if (acc.finite === false) reasons.push('nicht endlich');
  if (typeof decision === 'number' && decision < need) reasons.push(`Entscheidung ${decision} < ${need}`);
  if (task === 'embeddings' && acc.minCosine < 0.9999) reasons.push(`Kosinus ${acc.minCosine} < 0.9999`);
  const diff = acc.maxAbsLogitDiff ?? acc.maxAbsDiff;
  const absBound = f16 && deltaSim && !normalized;
  if (absBound && typeof diff === 'number' && diff > 3 * deltaSim) reasons.push(`Abweichung ${diff} > 3 x ${deltaSim}`);
  if (vsRef) {
    if (absBound && vsRef.maxAbsDiff > 3 * deltaSim) reasons.push(`Abweichung gegen CPU f32 ${vsRef.maxAbsDiff} > 3 x ${deltaSim}`);
    if (typeof vsRef.argmaxAgreement === 'number' && vsRef.argmaxAgreement < need) reasons.push(`Argmax gegen CPU f32 ${vsRef.argmaxAgreement}`);
    if (typeof vsRef.minCosine === 'number' && vsRef.minCosine < 0.9999) reasons.push(`Kosinus gegen CPU f32 ${vsRef.minCosine}`);
  }
  return { pass: reasons.length === 0, reasons };
}

// Geometric mean of the ratios new / ref minus 1 (the R8 step value, docs/R8_WORKORDER.md Festlegung 3).
export const geoChange = (rows) => Math.exp(rows.reduce((s, r) => s + Math.log(r.newMs / r.refMs), 0) / rows.length) - 1;

const fmt = (v, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d).replace('.', ',') : '');

// Standard sample size of the table (Festlegung 4); cells below it are marked as shorter samples.
export const standardN = (way, len) => (len === 'real' ? 200 : way === 'wasm' ? 50 : 100);

// One table cell of an engine from its run record, or the missing reason.
export function cellEntry(run, missing) {
  if (missing) return { status: missing };
  if (!run) return { status: 'fehlt: nicht gelaufen' };
  if (run.error) return { status: `fehlt: ${run.error.slice(0, 200)}` };
  return {
    status: 'ok', medianMs: run.medianMs, p95Ms: run.p95Ms, downloadMb: run.downloadMb ?? null,
    peakMemMib: run.peakMemMib ?? null, memNote: run.memNote ?? null, engineGzipB: run.engineGzipB ?? null,
    parity: run.parity ?? null, block: run.block ?? 1, n: run.n ?? null, config: run.config ?? null, runId: run.runId, file: run.file ?? null,
  };
}

// Table from the run records. `get(cell, engine)` returns the table run of an engine (or
// undefined), `mem(cell, engine)` the memory pass record, `miss(cell, engine)` the reason.
export function summarizeTable(cells, get, mem, miss) {
  return cells.map((c) => {
    const row = { ...c, short: modelOf(c.slug)?.short };
    for (const engine of ENGINES) {
      const m = miss(c, engine);
      const run = get(c, engine);
      const memRun = mem(c, engine);
      row[engine] = cellEntry(run ? { ...run, ...(memRun ? { peakMemMib: memRun.peakMemMib, memNote: memRun.memNote } : {}) } : run, m);
      if (row[engine].status === 'ok' && !memRun && !row[engine].memNote) row[engine].memNote = 'nicht gemessen';
      row[engine].shortSample = row[engine].status === 'ok' && typeof row[engine].n === 'number' && row[engine].n < standardN(c.way, c.len);
    }
    const k = row.kleinhirn; const o = row.ort;
    row.factor = k.status === 'ok' && o.status === 'ok' ? o.medianMs / k.medianMs : null;
    return row;
  });
}

const waysLabel = { 'webgpu-f16': 'WebGPU f16', 'webgpu-f32': 'WebGPU f32', wasm: 'WASM' };
const lenLabel = { 128: 'L128 voll', 512: 'L512 voll', real: 'echte Länge' };

function cellText(e, withMem) {
  if (e.status !== 'ok') return [e.status.replace(/\|/g, '/'), '', '', ''];
  return [
    `${fmt(e.medianMs)}${e.block > 1 ? '*' : ''}${e.shortSample ? '†' : ''}`, fmt(e.p95Ms),
    typeof e.downloadMb === 'number' ? fmt(e.downloadMb, 1) : '',
    withMem ? (typeof e.peakMemMib === 'number' ? fmt(e.peakMemMib, 0) : (e.memNote ?? '')) : '',
  ];
}

// r1-table.md: per browser and way one table, one row per model and length.
export function renderMarkdown(table, meta) {
  const out = [];
  out.push('# R1 kleinhirn-Bench: Tabelle', '');
  out.push(`Stand ${meta.date}, Commit ${meta.commit}, Screening-Qualität (ein Lauf je Zelle und Engine, nicht offiziell). Quelle: \`bench/results/r1-table.json\`, erzeugt von \`node bench/run-kbench.mjs summary\`.`, '');
  out.push('Median und p95 in ms je Aufruf, Download in MB, Speicher-Spitze in MiB (Chromium, Methode von `run-official.mjs`), Faktor ORT durch kleinhirn. Ein Stern am Median: Blockmessung (10 Aufrufe je Messwert, Festlegung 5), der p95 ist dort ein p95 über Blockmittel. Ein Kreuz (†): kürzere Stichprobe als die Norm (Firefox-WASM, n = 10 bei voller Länge und 50 echte Fälle statt 50 und 200). Parität kleinhirn: ok oder der Grund.', '');
  out.push(`Engine gzip -9: kleinhirn ${meta.gzip?.kleinhirn ?? '?'} B (\`dist/kleinhirn.js\`); ORT Web ${meta.gzip?.ort ?? '?'} B (JS und Wasm des Builds, im Mittel der Gewinner-Builds).`, '');
  const byBrowser = new Map();
  for (const r of table) {
    const k = `${r.browser}|${r.way}`;
    if (!byBrowser.has(k)) byBrowser.set(k, []);
    byBrowser.get(k).push(r);
  }
  let lastBrowser = null;
  for (const [k, rows] of byBrowser) {
    const [browser, way] = k.split('|');
    if (browser !== lastBrowser) { out.push(`## ${browser}`, ''); lastBrowser = browser; }
    out.push(`### ${browser}, ${waysLabel[way]}`, '');
    out.push('| Modell | Länge | kleinhirn Median | p95 | MB | MiB | ORT Median | p95 | MB | MiB | Faktor | Parität kleinhirn | ORT-Einstellung |');
    out.push('|---|---|---|---|---|---|---|---|---|---|---|---|---|');
    for (const r of rows) {
      const kc = cellText(r.kleinhirn, true); const oc = cellText(r.ort, true);
      const parity = r.kleinhirn.status === 'ok' ? (r.kleinhirn.parity?.pass ? 'ok' : (r.kleinhirn.parity?.reasons ?? ['offen']).join('; ')) : '';
      const cfg = r.ort.status === 'ok' && r.ort.config
        ? Object.entries(r.ort.config).filter(([, v]) => v !== null && v !== undefined && v !== '').map(([a, b]) => `${a}=${b}`).join(' ') : '';
      out.push(`| ${r.short} | ${lenLabel[r.len]} | ${kc.join(' | ')} | ${oc.join(' | ')} | ${fmt(r.factor)} | ${parity} | ${cfg} |`);
    }
    out.push('');
  }
  out.push('## iPhone', '', 'fehlt: R3, Gerät (alle Zellen).', '');
  return out.join('\n');
}

// ---------------------------------------------------------------- state, runs.tsv

export function readJson(file, fallback) {
  return existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : fallback;
}

export function writeJsonAtomic(file, data) {
  const tmp = `${file}.tmp-${process.pid}`;
  writeFileSync(tmp, `${JSON.stringify(data, null, 1)}\n`);
  renameSync(tmp, file);
}

export const COLS = [
  'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket', 'precision',
  'argmax_agreement', 'max_abs_logit_diff', 'max_abs_prob_diff', 'median_ms',
  'p95_ms', 'model_only_median_ms', 'load_ms', 'download_mb', 'gpu_mb',
  'peak_mem_mb', 'browser', 'adapter', 'kept', 'note',
];

export function appendRun(runId, fields) {
  const cells = COLS.map(() => '');
  for (const [k, v] of Object.entries({ date: new Date().toISOString().slice(0, 10), run_id: runId, kept: 'pending', ...fields })) {
    cells[COLS.indexOf(k)] = String(v).replace(/[\t\n]/g, ' ');
  }
  appendFileSync(RUNS, cells.join('\t') + '\n');
}

export function finishRun(runId, fields) {
  const lines = readFileSync(RUNS, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  if (idx < 0) throw new Error(`run ${runId} missing in runs.tsv`);
  const cells = lines[idx].split('\t');
  for (const [k, v] of Object.entries(fields)) cells[COLS.indexOf(k)] = String(v).replace(/[\t\n]/g, ' ');
  lines[idx] = cells.join('\t');
  writeFileSync(RUNS, lines.join('\n'));
}

// ---------------------------------------------------------------- environment

export class PauseError extends Error {}
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const load1 = () => loadavg()[0];
export const sh = (cmd, a, opts = {}) => execFileSync(cmd, a, { encoding: 'utf8', ...opts }).trim();

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

// Start rule: no foreign bench process, 1-minute load below MAX_LOAD (wait up to 30 minutes, then pause).
export async function checkEnvironment(ctx) {
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

export async function startVite(ctx) {
  let port = 5330;
  while (await portBusy(port)) port += 1;
  const proc = spawn(join(serveRoot, 'node_modules/.bin/vite'), ['--config', VITE_CONFIG, '--port', String(port), '--strictPort'],
    { cwd: serveRoot, stdio: 'ignore', detached: true });
  ctx.ownPids.add(proc.pid);
  ctx.vite = proc;
  for (let i = 0; i < 60; i += 1) {
    if (await portBusy(port)) { ctx.base = `http://localhost:${port}`; return { pid: proc.pid, port }; }
    await sleep(500);
  }
  throw new Error(`vite did not start on ${port}`);
}

export function stopVite(ctx) {
  if (!ctx.vite) return;
  try { process.kill(-ctx.vite.pid, 'SIGTERM'); } catch { /* gone */ }
  ctx.vite = null;
}

export function killStalePid(pid) {
  if (!pid) return;
  try { process.kill(pid, 0); process.kill(-pid, 'SIGTERM'); console.log(`stopped the own old vite (pid ${pid})`); } catch { /* gone */ }
}

export function buildIdOnDisk() {
  return readFileSync(join(serveRoot, 'dist/kleinhirn.js'), 'utf8').match(/buildId\s*:\s*["']([a-z0-9]+)["']/)?.[1];
}

export async function checkServedBuild(ctx) {
  const id = buildIdOnDisk();
  if (!id) throw new Error('no buildId in dist/kleinhirn.js');
  const served = await (await fetch(`${ctx.base}/dist/kleinhirn.js`)).text();
  if (!served.includes(id)) throw new Error(`vite serves a stale bundle (buildId ${id})`);
  return id;
}

export const bundleBytes = () => {
  const p = join(serveRoot, 'dist/kleinhirn.js');
  return { raw: statSync(p).size, gzip9: gzipBytes(p) };
};

// ---------------------------------------------------------------- browsers

// Resolves to { label, version, goto(url), state(), result(), close() }.
export async function openBrowser(name, globalName = 'khKbenchResult') {
  if (name === 'safari') return openSafari(globalName);
  if (name === 'firefox-reg') return openFirefoxRegular();
  const type = { chromium, webkit, firefox }[name];
  if (!type) throw new Error(`unknown browser ${name}`);
  const browser = await type.launch({ headless: false });
  const page = await browser.newPage();
  const crash = { error: null };
  page.on('crash', () => { crash.error = 'page crashed'; });
  page.on('pageerror', (e) => { crash.pageError = String(e).slice(0, 300); });
  return {
    label: `${name}-${browser.version()}`, version: browser.version(), rootProcess: true,
    goto: (url) => page.goto(url),
    async state() {
      if (crash.error) return { done: true, error: crash.error };
      try { return await page.evaluate((g) => { const r = window[g]; return r ? { stage: r.stage, done: r.done === true } : { stage: 'none', done: false }; }, globalName); } catch (e) { return { stage: 'nav', done: false, note: String(e).slice(0, 80) }; }
    },
    result: () => page.evaluate((g) => window[g], globalName),
    close: () => browser.close(),
  };
}

// A port the OS reports free; a random pick can land on another app's listener
// (Universal Audio listens on 4710, 4720, 4793), which then answers the WebDriver calls.
function freePort() {
  return new Promise((res, rej) => {
    const srv = createServer();
    srv.once('error', rej);
    srv.listen(0, '127.0.0.1', () => { const { port } = srv.address(); srv.close(() => res(port)); });
  });
}

// Safari through safaridriver, raw WebDriver over HTTP (pattern of bench/run-site.mjs).
async function openSafari(globalName) {
  const dPort = await freePort();
  const driver = spawn('safaridriver', ['-p', String(dPort)], { stdio: 'ignore' });
  const root = `http://127.0.0.1:${dPort}`;
  const call = async (method, path, body) => {
    const res = await fetch(`${root}${path}`, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(`webdriver ${method} ${path}: ${res.status} ${JSON.stringify(json.value)}`);
    return json.value;
  };
  let sid = null;
  const cleanup = async () => {
    if (sid) { try { await call('DELETE', `/session/${sid}`); } catch { /* gone */ } sid = null; }
    driver.kill();
  };
  try {
    let ready = false;
    for (let i = 0; i < 100 && !ready; i += 1) {
      try { ready = (await (await fetch(`${root}/status`)).json()).value?.ready === true; } catch { /* not up yet */ }
      if (!ready) await sleep(100);
    }
    if (!ready) throw new Error(`safaridriver on port ${dPort} did not report ready`);
    const created = await call('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari' } } });
    sid = created.sessionId;
    const version = created.capabilities.browserVersion;
    const exec = (script) => call('POST', `/session/${sid}/execute/sync`, { script, args: [] });
    return {
      label: `safari-${version}`, version, rootProcess: false,
      async goto(url) { await call('POST', `/session/${sid}/url`, { url }); },
      async state() {
        try { return JSON.parse(await exec(`var r = window.${globalName}; return JSON.stringify(r ? { stage: r.stage, done: r.done === true } : { stage: 'none', done: false });`)); } catch (e) { return { stage: 'nav', done: false, note: String(e).slice(0, 80) }; }
      },
      result: async () => JSON.parse(await exec(`return JSON.stringify(window.${globalName})`)),
      close: cleanup,
    };
  } catch (e) {
    await cleanup();
    throw e;
  }
}

// ---------------------------------------------------------------- regular Firefox (R8)

export const FIREFOX_APP = process.env.KBENCH_FIREFOX_APP ?? join(ROOT, 'models/browsers/Firefox.app');

const FIREFOX_PREFS = [
  ['browser.shell.checkDefaultBrowser', false], ['browser.startup.homepage_override.mstone', 'ignore'],
  ['browser.aboutwelcome.enabled', false], ['browser.startup.page', 0], ['startup.homepage_welcome_url', ''],
  ['datareporting.policy.dataSubmissionEnabled', false], ['datareporting.healthreport.uploadEnabled', false],
  ['toolkit.telemetry.reportingpolicy.firstRun', false], ['toolkit.telemetry.enabled', false],
  ['app.update.auto', false], ['app.update.enabled', false], ['browser.sessionstore.resume_from_crash', false],
  ['browser.tabs.warnOnClose', false], ['browser.warnOnQuit', false], ['dom.disable_open_during_load', false],
];

// A regular Firefox (R8 Festlegung 9): started directly with a fresh profile, no Playwright, no
// WebDriver, so WASM compiles with the optimizing tier (FINDINGS 54). The page gets
// ?khpost=<collector> (bench/kbench/post.ts) and posts its stage and result; this process
// collects them. A result with navigator.webdriver true is an error.
async function openFirefoxRegular() {
  const bin = join(FIREFOX_APP, 'Contents/MacOS/firefox');
  if (!existsSync(bin)) throw new Error(`no Firefox at ${FIREFOX_APP}`);
  const version = sh('plutil', ['-extract', 'CFBundleShortVersionString', 'raw', join(FIREFOX_APP, 'Contents/Info.plist')]);
  const token = Math.random().toString(36).slice(2);
  let stage = { stage: 'none', done: false };
  let result = null;
  const server = createHttpServer((req, res) => {
    const cors = { 'Access-Control-Allow-Origin': '*', 'Connection': 'close' };
    if (req.method !== 'POST' || req.url !== `/${token}`) { res.writeHead(404, cors).end(); return; }
    let body = '';
    req.on('data', (d) => { body += d; });
    req.on('end', () => {
      res.writeHead(200, cors).end('ok');
      try {
        const msg = JSON.parse(body);
        if (msg.done) {
          result = msg;
          stage = msg.webdriver === true ? { stage: 'error', done: true, error: 'navigator.webdriver is true' } : { stage: msg.stage, done: true };
        } else stage = { stage: msg.stage, done: false };
      } catch (e) { stage = { stage: 'error', done: true, error: `bad post: ${String(e).slice(0, 100)}` }; }
    });
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const post = `http://127.0.0.1:${server.address().port}/${token}`;
  const profile = mkdtempSync(join(tmpdir(), 'kh-ff-reg-'));
  writeFileSync(join(profile, 'user.js'), FIREFOX_PREFS.map(([k, v]) => `user_pref(${JSON.stringify(k)}, ${JSON.stringify(v)});`).join('\n') + '\n');
  let ff = null;
  const close = async () => {
    if (ff && ff.exitCode === null) {
      try { process.kill(-ff.pid, 'SIGTERM'); } catch { /* gone */ }
      for (let i = 0; i < 50 && ff.exitCode === null; i += 1) await sleep(100);
      if (ff.exitCode === null) { try { process.kill(-ff.pid, 'SIGKILL'); } catch { /* gone */ } }
    }
    server.close();
    await sleep(500);
    rmSync(profile, { recursive: true, force: true });
  };
  return {
    label: `firefox-reg-${version}`, version, rootProcess: true,
    async goto(url) {
      const u = `${url}${url.includes('?') ? '&' : '?'}khpost=${encodeURIComponent(post)}`;
      ff = spawn(bin, ['--no-remote', '--new-instance', '--profile', profile, u], { stdio: 'ignore', detached: true });
      ff.on('exit', () => { if (!stage.done) stage = { stage: 'error', done: true, error: `firefox exited (${ff.exitCode})` }; });
    },
    async state() {
      return stage.error ? { stage: stage.stage, done: true, error: stage.error } : { stage: stage.stage, done: stage.done };
    },
    async result() {
      return stage.error ? { ...(result ?? {}), error: stage.error, done: true } : result;
    },
    close,
  };
}
