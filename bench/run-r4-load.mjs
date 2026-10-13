// R4 Stufe 4 load bench runner (docs/R4_WORKORDER.md, G2, G5): time to the first result per model,
// mode and network profile, Chromium with WebGPU f16, one fresh browser and context per run.
// Page: bench/r4-load.html and bench/r4-load.ts (window.khR4Load).
//
// Usage: node bench/run-r4-load.mjs [--models minilm,...] [--modes pre,full,rows,progressive]
//        [--profiles local,50m,10m] [--reps n] [--port p] [--base <worktree>] [--tag t] [--calls n] [--warm n]
//        node bench/run-r4-load.mjs --summary [--tag t]
// Modes: pre serves the pre-R4 worktree (--base, default ../kleinhirn-r4-base at adac8a0) and loads
// without the words option; full, rows and progressive serve this worktree with words=<mode>.
// Profiles: local (no throttle), 50m (50 Mbit/s, 20 ms RTT), 10m (10 Mbit/s, 100 ms RTT), per CDP.
// Every run: a runs.tsv line before (change r4-load-<mode>-<profile>), finished after with load_ms
// (ttfr), download_mb and the result file; result bench/results/<tag>-r4-load-<model>-<mode>-<profile>-r<rep>.json;
// state data/r4-load-state.json, so a rerun skips finished runs. Order per model and profile: the
// modes forward in odd reps, reversed in even reps.
// Guards: refuse to start above a 1-minute load of 6; wait below 4 before each run (checkEnvironment);
// no foreign bench process; stop when the summed RSS of "Chrome for Testing" exceeds 6000 MB (exit 6).
// Exit codes: 3 pause (load or foreign processes), 6 Chrome memory guard.

import { chromium } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { loadavg } from 'node:os';
import { join, resolve } from 'node:path';
import {
  MODELS, PauseError, ROOT, RESULTS, appendRun, checkEnvironment, finishRun, readJson, sleep, writeJsonAtomic,
} from './kbench/runner-lib.mjs';

const STATE_FILE = join(ROOT, 'data/r4-load-state.json');
const PROFILES = {
  local: null,
  '50m': { latency: 20, downloadThroughput: 50e6 / 8, uploadThroughput: 20e6 / 8 },
  '10m': { latency: 100, downloadThroughput: 10e6 / 8, uploadThroughput: 5e6 / 8 },
};
const ALL_MODES = ['pre', 'full', 'rows', 'progressive'];
const MAX_CHROME_MB = 6000;
const MAX_START_LOAD = 6;
const RUN_TIMEOUT_MS = 30 * 60000;

const args = process.argv.slice(2);
const arg = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const list = (k, d) => (arg(k, null) ? arg(k, null).split(',') : d);
const TAG = arg('--tag', 'r4');
const PORT = Number(arg('--port', 5246));
const REPS = Number(arg('--reps', 3));
const CALLS = Number(arg('--calls', 20));
const WARM = Number(arg('--warm', 0)); // rounds over the same cases with every row in memory (G6)
const PRECISION = arg('--precision', 'f16');
const BASE = resolve(arg('--base', join(ROOT, '..', 'kleinhirn-r4-base')));
const MODES = list('--modes', ALL_MODES);
const PROFILE_KEYS = list('--profiles', Object.keys(PROFILES));
const SHORT = Object.fromEntries(MODELS.map((m) => [m.short.toLowerCase(), m]));
const MODEL_LIST = list('--models', null)?.map((s) => SHORT[s.toLowerCase()] ?? MODELS.find((m) => m.slug === s))
  ?? MODELS;

for (const m of MODES) if (!ALL_MODES.includes(m)) throw new Error(`mode ${m}: none of ${ALL_MODES.join(', ')}`);
for (const p of PROFILE_KEYS) if (!(p in PROFILES)) throw new Error(`profile ${p}: none of ${Object.keys(PROFILES).join(', ')}`);
if (MODEL_LIST.some((m) => !m)) throw new Error('unknown model name, see MODELS in bench/kbench/runner-lib.mjs');

const git = (dir) => execFileSync('git', ['-C', dir, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
const relFile = (f) => f.replace(`${ROOT}/`, '');

// ---------------------------------------------------------------- guards

function chromeMb() {
  const out = execFileSync('ps', ['-Ao', 'rss,command'], { encoding: 'utf8' });
  let kb = 0;
  for (const line of out.split('\n')) {
    if (line.includes('Chrome for Testing') && !line.includes('grep')) kb += Number(line.trim().split(/\s+/)[0]) || 0;
  }
  return Math.round(kb / 1024);
}

// Other sessions' browser runs of the same family (the workorder's pgrep rule); own pid excluded.
function foreignRuns() {
  let r = '';
  try { r = execFileSync('pgrep', ['-fl', 'run-k28-parity|run-bits|r4_g1|run-kbench|run-r4-load'], { encoding: 'utf8' }); } catch { /* none */ }
  return r.split('\n').filter((l) => l && !l.startsWith(`${process.pid} `)).map((l) => l.slice(0, 160));
}

const ctx = { ownPids: new Set() };
let current = null;   // the open browser of the run in progress
const vites = [];     // { proc, port }

function stopVites() {
  for (const v of vites) { try { process.kill(-v.proc.pid, 'SIGTERM'); } catch { /* gone */ } }
  vites.length = 0;
}

setInterval(async () => {
  const mb = chromeMb();
  if (mb <= MAX_CHROME_MB) return;
  console.log(`GUARD chrome ${mb} MB > ${MAX_CHROME_MB} MB, stopping`);
  await current?.close().catch(() => {});
  stopVites();
  process.exit(6);
}, 2000).unref();

// ---------------------------------------------------------------- servers

async function startVite(dir, port) {
  const proc = spawn(join(dir, 'node_modules/.bin/vite'), ['--port', String(port), '--strictPort'],
    { cwd: dir, stdio: 'ignore', detached: true });
  ctx.ownPids.add(proc.pid);
  vites.push({ proc, port });
  for (let i = 0; i < 120; i += 1) {
    try { if ((await fetch(`http://localhost:${port}/bench/r4-load.html`)).ok) return; } catch { /* not yet */ }
    await sleep(500);
  }
  throw new Error(`vite did not start in ${dir} on ${port}`);
}

// The pre-R4 worktree serves the page copied from here; its copy stays untracked there.
function copyPage(dir) {
  if (dir === ROOT) return;
  mkdirSync(join(dir, 'bench'), { recursive: true });
  for (const f of ['r4-load.html', 'r4-load.ts']) copyFileSync(join(ROOT, 'bench', f), join(dir, 'bench', f));
}

// ---------------------------------------------------------------- one run

function modeServe(mode) {
  return mode === 'pre' ? { dir: BASE, port: PORT + 1 } : { dir: ROOT, port: PORT };
}

async function runOne({ model, mode, profile, rep, state }) {
  const key = `${TAG}|${model.slug}|${mode}|${profile}|${rep}`;
  const file = join(RESULTS, `${TAG}-r4-load-${model.slug}-${mode}-${profile}-r${rep}.json`);
  if (state.runs[key] && existsSync(file)) { console.log(`skip ${key} (done)`); return; }

  const { dir, port } = modeServe(mode);
  const commit = git(dir);
  const runId = `${TAG}-r4-load-${model.short}-${mode}-${profile}-r${rep}`;
  const change = `r4-load-${mode}-${profile}`;
  const words = mode === 'pre' ? null : mode;
  const url = `http://localhost:${port}/bench/r4-load.html?model=${model.slug}&precision=${PRECISION}&calls=${CALLS}&warm=${WARM}`
    + (words ? `&words=${words}` : '');

  await checkEnvironment(ctx);
  const env = { loadStart: Number(loadavg()[0].toFixed(2)) };
  appendRun(runId, { change, commit, engine: 'kleinhirn', model: model.slug, precision: PRECISION, browser: 'chromium', note: `load bench ${mode} ${profile}` });

  let doc;
  const browser = await chromium.launch({ headless: false });
  current = browser;
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    const cdp = await context.newCDPSession(page);
    await cdp.send('Network.enable');
    if (PROFILES[profile]) await cdp.send('Network.emulateNetworkConditions', { offline: false, ...PROFILES[profile] });
    await page.goto(url);
    await page.waitForFunction(() => window.khR4Load?.done === true, null, { timeout: RUN_TIMEOUT_MS });
    const r = await page.evaluate(() => window.khR4Load);
    doc = { result: r, error: r.error ?? null };
  } catch (e) {
    doc = { result: null, error: String(e.message ?? e).slice(0, 400) };
  } finally {
    doc.browser = browser.version();
    await browser.close().catch(() => {});
    current = null;
  }

  const bytes = doc.result?.bytes;
  const meta = {
    date: new Date().toISOString(), tag: TAG, model: model.slug, short: model.short, mode, profile, rep, precision: PRECISION,
    commit, serve: dir, url, calls: CALLS, warm: WARM, env, throttle: PROFILES[profile], ...doc,
  };
  const rel = relFile(file);
  writeJsonAtomic(file, meta);
  const mb = bytes ? (bytes.total / 1e6).toFixed(2) : '';
  finishRun(runId, {
    load_ms: doc.result?.ttfrMs !== undefined ? doc.result.ttfrMs.toFixed(1) : '',
    download_mb: mb,
    browser: `chromium-${doc.browser}`,
    kept: meta.error ? 'error' : 'done',
    note: `${meta.error ? `error ${meta.error.slice(0, 120)}` : `ttfr ${doc.result.ttfrMs.toFixed(0)} ms`} ${rel}`,
  });
  state.runs[key] = { file: rel, ttfrMs: doc.result?.ttfrMs ?? null, error: meta.error };
  writeJsonAtomic(STATE_FILE, state);
  console.log(`${key}: ${meta.error ? `error ${meta.error.slice(0, 120)}` : `ttfr ${doc.result.ttfrMs.toFixed(0)} ms, load ${doc.result.loadMs.toFixed(0)} ms, ${mb} MB`}`);
}

// ---------------------------------------------------------------- summary

const median = (xs) => {
  const s = xs.filter((x) => typeof x === 'number' && Number.isFinite(x)).sort((a, b) => a - b);
  if (!s.length) return null;
  const h = Math.floor(s.length / 2);
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
};
const r1 = (x) => (x === null ? '-' : Math.round(x).toString());

function summary() {
  const state = readJson(STATE_FILE, { runs: {} });
  const groups = new Map();
  for (const [key, entry] of Object.entries(state.runs)) {
    if (!key.startsWith(`${TAG}|`) || entry.error) continue;
    const [, slug, mode, profile] = key.split('|');
    const doc = readJson(join(ROOT, entry.file), null);
    if (!doc?.result) continue;
    const g = `${slug}|${mode}|${profile}`;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(doc);
  }
  const rows = [];
  for (const [g, docs] of groups) {
    const [slug, mode, profile] = g.split('|');
    const res = docs.map((d) => d.result);
    const calls = res.flatMap((r) => r.callMs ?? []);
    const warm = res.flatMap((r) => r.warmMs ?? []);
    const tok = res.map((r) => Object.entries(r.loadTiming ?? {}).filter(([k]) => /token/i.test(k)).reduce((s, [, v]) => s + v, 0)).filter((v) => v > 0);
    const w = res[0].words ?? {};
    rows.push({
      model: MODELS.find((m) => m.slug === slug)?.short ?? slug, mode, profile, reps: docs.length,
      ttfrMs: median(res.map((r) => r.ttfrMs)), loadMs: median(res.map((r) => r.loadMs)),
      firstCallMs: median(res.map((r) => r.firstCallMs)),
      callMedianMs: median(calls), callMaxMs: calls.length ? Math.max(...calls) : null,
      warmMedianMs: warm.length ? median(warm) : null,
      mbModels: median(res.map((r) => (r.bytes?.models ?? 0) / 1e6)), mbEngine: median(res.map((r) => (r.bytes?.engine ?? 0) / 1e6)),
      tokenizerMs: tok.length ? median(tok) : null,
      words: { mode: w.requested ?? '-', rows: w.rows ?? '-', requests: w.requests ?? '-', bytes: w.bytes ?? '-' },
    });
  }
  rows.sort((a, b) => a.model.localeCompare(b.model) || a.profile.localeCompare(b.profile) || a.mode.localeCompare(b.mode));
  const head = '| model | mode | profile | reps | ttfr ms | load ms | first call ms | call median ms | call max ms | warm median ms | MB models | MB engine | tokenizer ms | words (mode, rows, requests, bytes) |';
  const md = [`# R4 load bench, tag ${TAG}`, '', head, `|${'---|'.repeat(14)}`];
  for (const r of rows) {
    md.push(`| ${r.model} | ${r.mode} | ${r.profile} | ${r.reps} | ${r1(r.ttfrMs)} | ${r1(r.loadMs)} | ${r1(r.firstCallMs)} | ${r1(r.callMedianMs)} | ${r1(r.callMaxMs)} | ${r.warmMedianMs === null ? "-" : r.warmMedianMs.toFixed(2)} | ${r.mbModels === null ? '-' : r.mbModels.toFixed(2)} | ${r.mbEngine === null ? '-' : r.mbEngine.toFixed(2)} | ${r1(r.tokenizerMs)} | ${r.words.mode}, ${r.words.rows}, ${r.words.requests}, ${r.words.bytes} |`);
  }
  md.push('', 'ttfr, load, first call: median of reps. Call median and max: pooled over all reps of the cell. MB: decimal, from Resource Timing.');
  writeJsonAtomic(join(RESULTS, `${TAG}-r4-load-summary.json`), rows);
  const mdFile = join(RESULTS, `${TAG}-r4-load-summary.md`);
  writeFileSync(`${mdFile}.tmp`, `${md.join('\n')}\n`);
  renameSync(`${mdFile}.tmp`, mdFile);
  console.log(md.join('\n'));
}

// ---------------------------------------------------------------- main

async function main() {
  if (args.includes('--summary')) { summary(); return; }
  const load = loadavg()[0];
  if (load > MAX_START_LOAD) {
    console.log(`refusing to start: 1-minute load ${load.toFixed(2)} > ${MAX_START_LOAD}`);
    process.exit(3);
  }
  const foreign = foreignRuns();
  if (foreign.length) {
    console.log(`refusing to start, foreign runs:\n${foreign.join('\n')}`);
    process.exit(3);
  }
  mkdirSync(RESULTS, { recursive: true });
  mkdirSync(join(ROOT, 'data'), { recursive: true });
  const state = readJson(STATE_FILE, { runs: {} });

  if (MODES.includes('pre')) {
    if (!existsSync(join(BASE, 'node_modules/.bin/vite'))) throw new Error(`no vite in ${BASE}`);
    copyPage(BASE);
    await startVite(BASE, PORT + 1);
  }
  if (MODES.some((m) => m !== 'pre')) {
    copyPage(ROOT);
    await startVite(ROOT, PORT);
  }
  console.log(`serving: ${MODES.includes('pre') ? `pre ${BASE} on ${PORT + 1}; ` : ''}this worktree on ${PORT} (${git(ROOT)})`);

  try {
    for (const model of MODEL_LIST) {
      for (const profile of PROFILE_KEYS) {
        for (let rep = 1; rep <= REPS; rep += 1) {
          const order = rep % 2 === 0 ? [...MODES].reverse() : MODES;
          for (const mode of order) await runOne({ model, mode, profile, rep, state });
        }
      }
    }
  } catch (e) {
    if (e instanceof PauseError) { console.log(`pause: ${e.message}`); stopVites(); process.exit(3); }
    throw e;
  } finally {
    stopVites();
  }
}

main().catch((e) => { console.error(e); stopVites(); process.exit(1); });
