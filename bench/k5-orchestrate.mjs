// K5 official device matrix: runs the rows of the matrix interleaved
// (rep 1 of every row, then rep 2, then rep 3) plus the control pair, all from
// the official worktree (start this script there) with results in the main
// checkout.
//
// Usage (cwd = official worktree, site/dist built there):
//   node bench/k5-orchestrate.mjs --out-root <main checkout> [--reps 3]
//        [--rows chromium,brave,...] [--firefox-stages f16,f32[,wasm]]
//        [--no-control] [--only-control]
//
// Load rule: a job starts only when the 1-minute load is below MAX_LOAD (4).
// The orchestrator polls every 60 s, prints the load every 30 min, stops after
// BUDGET_H hours of waiting in total, and stops after STALL_H hours of
// continuous load above MAX_LOAD with the top 5 processes by CPU. The runner
// checks the load again right before the browser starts and records it.
//
// State: <out-root>/bench/results/k5-official-state.json lists finished jobs;
// a restart skips them.

import { execFileSync, spawn } from 'node:child_process';
import { loadavg } from 'node:os';
import { existsSync, readdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const MAX_LOAD = 4;
const BUDGET_H = 6;
const STALL_H = 2;
const arg = (name, dflt = null) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const OUT = resolve(arg('out-root', '.'));
const REPS = Number(arg('reps', 3));
const DEVICE = 'MacBook Pro M1 Pro 16 GB (official)';
const STATE = resolve(OUT, 'bench/results/k5-official-state.json');
const fStages = arg('firefox-stages', 'f16,f32');

const ROWS = [
  { id: 'chromium', browser: 'chromium', stages: 'f16,f32,wasm', timeoutMin: 90 },
  { id: 'brave', browser: 'brave', stages: 'f16,f32,wasm', timeoutMin: 90 },
  { id: 'webkit', browser: 'webkit', stages: 'f16,f32,wasm', timeoutMin: 120 },
  { id: 'safari', browser: 'safari', validateSafari: true, keepAwake: true, stages: 'f16,f32,wasm', timeoutMin: 240, extra: ['--safari-mode', 'local', '--poll-s', '10'] },
  { id: 'firefox', browser: 'firefox', stages: fStages, timeoutMin: 90 },
  { id: 'chromium-nogpu', browser: 'chromium-nogpu', stages: 'wasm', timeoutMin: 90 },
];
const only = arg('rows') ? new Set(arg('rows').split(',')) : null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const load1 = () => loadavg()[0];
const state = existsSync(STATE) ? JSON.parse(readFileSync(STATE, 'utf8')) : { done: {}, waitMs: 0, starts: [] };
const save = () => writeFileSync(STATE, JSON.stringify(state, null, 1));

function top5() {
  return execFileSync('ps', ['-axo', 'pcpu=,pid=,command='], { encoding: 'utf8', maxBuffer: 64 << 20 })
    .split('\n').map((l) => l.trim()).filter(Boolean)
    .map((l) => ({ cpu: Number(l.split(/\s+/)[0]), line: l.slice(0, 150) }))
    .sort((a, b) => b.cpu - a.cpu).slice(0, 5).map((x) => x.line);
}

let aboveSince = null;
let lastPrint = 0;
async function waitForLoad(job) {
  for (;;) {
    const l = load1();
    if (l < MAX_LOAD) { aboveSince = null; return l; }
    const now = Date.now();
    aboveSince ??= now;
    if (now - lastPrint > 30 * 60000 || lastPrint === 0) {
      lastPrint = now;
      console.log(`${new Date().toISOString()} waiting for ${job}: 1-min load ${l.toFixed(2)}, waited ${(state.waitMs / 3600000).toFixed(2)} h in total`);
    }
    if (now - aboveSince > STALL_H * 3600000) {
      console.log(`STOP: load above ${MAX_LOAD} for ${STALL_H} h straight. Top 5 by CPU:\n${top5().join('\n')}`);
      save();
      process.exit(3);
    }
    if (state.waitMs > BUDGET_H * 3600000) {
      console.log(`STOP: waited more than ${BUDGET_H} h in total. Top 5 by CPU:\n${top5().join('\n')}`);
      save();
      process.exit(4);
    }
    await sleep(60000);
    state.waitMs += 60000;
    save();
  }
}

function spawnRunner(cmd, args, env = {}, keepAwakeMin = 0) {
  return new Promise((res) => {
    const awake = keepAwakeMin ? spawn('caffeinate', ['-disu', '-t', String(keepAwakeMin * 60)], { stdio: 'ignore' }) : null;
    const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, ...env } });
    let text = '';
    const take = (d) => { text += d; process.stdout.write(d); };
    child.stdout.on('data', take);
    child.stderr.on('data', take);
    child.on('close', (code) => { awake?.kill(); res({ code, text }); });
  });
}

async function job(id, cmd, args, env, keepAwakeMin = 0) {
  if (state.done[id]) { console.log(`skip ${id} (done)`); return; }
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const l = await waitForLoad(id);
    console.log(`${new Date().toISOString()} start ${id} (attempt ${attempt}), load ${l.toFixed(2)}`);
    state.starts.push({ id, attempt, at: new Date().toISOString(), load1: l });
    const t0 = Date.now();
    const { code, text } = await spawnRunner(cmd, args, env, keepAwakeMin);
    if (code === 0) {
      state.done[id] = { at: new Date().toISOString(), wallMin: Number(((Date.now() - t0) / 60000).toFixed(1)) };
      save();
      return;
    }
    console.log(`${id} failed with code ${code}`);
    const loadRefusal = /still above/.test(text);
    save();
    if (!loadRefusal && attempt >= 2) break;
  }
  state.done[id] = { failed: true, at: new Date().toISOString() };
  save();
}

// Safari throttles a window on a locked screen with a sleeping display to
// bursts (wasm calls 4x slower, single calls stall for minutes; K5 diagnosis c).
// Safari jobs therefore run under `caffeinate -disu` (a power assertion that
// wakes the display and ends with the job) and every finished Safari rep is
// checked for stalls or a throttled series; a bad rep is moved to
// k5-rejected-* and run again (up to 3 attempts, each recorded in the state).
const RESULTS = resolve(OUT, 'bench/results');
function safariRepProblem(rep) {
  const name = readdirSync(RESULTS).filter((f) => new RegExp(`^k5-official-safari-\\d+T\\d+-r${rep}\\.json$`).test(f)).sort().pop();
  if (!name) return { file: null, problem: 'no result file' };
  const { result } = JSON.parse(readFileSync(resolve(RESULTS, name), 'utf8'));
  if (!result || result.error) return { file: name, problem: `page error ${result?.error}` };
  const problems = [];
  for (const st of result.stages) {
    if (!st.ok) continue;
    for (const b of ['L128', 'L256']) {
      for (const kind of ['e2e', 'modelOnly']) {
        const l = st.latency[b][kind];
        if (l.maxMs > 25 * l.medianMs && l.maxMs > 500) problems.push(`${st.name} ${b} ${kind} stall: max ${l.maxMs.toFixed(0)} ms vs median ${l.medianMs.toFixed(1)}`);
      }
    }
    const l128 = st.latency.L128.modelOnly.medianMs;
    const l256 = st.latency.L256.modelOnly.medianMs;
    if (st.name === 'f16' && l128 > 40) problems.push(`f16 L128 model-only median ${l128.toFixed(1)} ms (clean runs: about 20)`);
    if (st.name === 'wasm' && l256 > 1500) problems.push(`wasm L256 model-only median ${l256.toFixed(0)} ms (clean runs: about 700)`);
  }
  return { file: name, problem: problems.length ? problems.join('; ') : null };
}

const siteRun = async (row, rep) => {
  const id = `${row.id}-r${rep}`;
  if (!row.validateSafari) return siteRunJob(row, rep);
  for (let attempt = 1; attempt <= 3 && !(state.done[id] && !state.done[id].failed); attempt += 1) {
    delete state.done[id];
    await siteRunJob(row, rep);
    if (state.done[id]?.failed) continue;
    const { file, problem } = safariRepProblem(rep);
    if (!problem) return;
    console.log(`${id} attempt ${attempt} rejected: ${problem}`);
    (state.rejected ??= []).push({ id, attempt, file, problem });
    if (file) for (const ext of ['.json', '.result.json']) renameSync(resolve(RESULTS, file.replace(/\.json$/, ext)), resolve(RESULTS, file.replace(/^k5-official/, 'k5-rejected').replace(/\.json$/, ext)));
    delete state.done[id];
    save();
  }
  if (!state.done[id]) { state.done[id] = { failed: true, notRun: 'three attempts rejected or failed' }; save(); }
};
const siteRunJob = (row, rep) => job(
  `${row.id}-r${rep}`, 'node',
  ['bench/run-site.mjs', '--browser', row.browser, '--stages', row.stages, '--weights', 'local',
    '--change', 'k5-official', '--device', DEVICE, '--max-load', String(MAX_LOAD), '--wait-min', '2',
    '--out-root', OUT, '--rep-index', String(rep), '--timeout-min', String(row.timeoutMin), ...(row.extra ?? [])],
  {}, row.keepAwake ? row.timeoutMin : 0,
);

async function control() {
  // ABAB: page (chromium, stages=f16) against the K3/K20 runner, 3 reps each.
  const vite = spawn('npx', ['vite', '--port', '5199', '--strictPort'], { stdio: 'ignore' });
  try {
    for (let i = 0; i < 30; i += 1) {
      try { await fetch('http://localhost:5199/', { signal: AbortSignal.timeout(1000) }); break; } catch { await sleep(1000); }
    }
    for (let rep = 1; rep <= REPS; rep += 1) {
      await job(`control-page-r${rep}`, 'node', ['bench/run-site.mjs', '--browser', 'chromium', '--stages', 'f16', '--weights', 'local',
        '--change', 'k5-control-page', '--device', DEVICE, '--max-load', String(MAX_LOAD), '--wait-min', '2',
        '--out-root', OUT, '--rep-index', String(rep), '--timeout-min', '30']);
      await job(`control-runner-r${rep}`, 'node', ['bench/run-kleinhirn.mjs', 'small-upstream', 'f16', 'k5-control-runner'], { KH_OUT_ROOT: OUT });
    }
  } finally {
    vite.kill();
  }
}

async function main() {
  if (!process.argv.includes('--only-control')) {
    for (let rep = 1; rep <= REPS; rep += 1) {
      for (const row of ROWS) {
        if (only && !only.has(row.id)) continue;
        await siteRun(row, rep);
      }
    }
  }
  if (!process.argv.includes('--no-control')) await control();
  console.log(`done. total load wait ${(state.waitMs / 60000).toFixed(0)} min; failed jobs: ${Object.entries(state.done).filter(([, v]) => v.failed).map(([k]) => k).join(', ') || 'none'}`);
}

await main();
