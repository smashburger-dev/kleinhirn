// K27: A/B of two engine commits on the K28.8 kleinhirn page (bench/k28-latency.html, n = 300),
// interleaved A B A B per cell, fresh Chromium per run, start only at 1-minute load below 4.
// A is served by ../kleinhirn-official (checked out at commit A and built by this script), B by
// the k28 worktree as it is (build it first); both with bench/k28/vite.k28-8.config.ts.
// One runs.tsv line per run (change k27-ab). Result: bench/results/k27-ab-<A>-<B>.json
// Usage: node bench/run-k27-ab.mjs <commitA> <slug>:<L>[,<slug>:<L>] [reps=3]

import { chromium } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { appendFileSync, writeFileSync } from 'node:fs';
import { createConnection } from 'node:net';
import { loadavg } from 'node:os';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const K28 = resolve(fileURLToPath(new URL('..', import.meta.url)));
const OFFICIAL = resolve(K28, '../kleinhirn-official');
const [commitA, cellsArg, repsArg = '3'] = process.argv.slice(2);
const cells = cellsArg.split(',').map((c) => { const [slug, L] = c.split(':'); return { slug, L: Number(L) }; });
const reps = Number(repsArg);
const sh = (cmd, a, o = {}) => execFileSync(cmd, a, { encoding: 'utf8', ...o }).trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const busy = (port) => new Promise((res) => { const s = createConnection({ port, host: 'localhost' }); s.on('connect', () => { s.destroy(); res(true); }); s.on('error', () => res(false)); });

async function vite(cwd, port) {
  const p = spawn(resolve(cwd, 'node_modules/.bin/vite'), ['--config', 'bench/k28/vite.k28-8.config.ts', '--port', String(port), '--strictPort'],
    { cwd, stdio: 'ignore', detached: true });
  for (let i = 0; i < 60 && !(await busy(port)); i += 1) await sleep(500);
  return p;
}

const commitB = sh('git', ['-C', K28, 'rev-parse', '--short', 'HEAD']);
const original = sh('git', ['-C', OFFICIAL, 'rev-parse', 'HEAD']);
if (sh('git', ['-C', OFFICIAL, 'status', '--porcelain', '--', 'src'])) throw new Error('official worktree src not clean');
sh('git', ['-C', OFFICIAL, 'checkout', '-q', '--detach', commitA]);
execFileSync('npm', ['run', 'build'], { cwd: OFFICIAL, stdio: 'ignore' });
const servers = [await vite(OFFICIAL, 5330), await vite(K28, 5331)];
const out = { commitA, commitB, reps, cells: {} };
try {
  for (const { slug, L } of cells) {
    const key = `${slug}|L${L}`;
    out.cells[key] = { A: [], B: [] };
    for (let r = 0; r < reps; r += 1) {
      for (const [side, port] of [['A', 5330], ['B', 5331]]) {
        while (loadavg()[0] >= 4) await sleep(15000);
        const loadStart = loadavg()[0];
        const runId = `k27-ab-${side}-${slug}-L${L}-${Date.now()}`;
        appendFileSync(resolve(K28, 'data/hillclimb/runs.tsv'), [new Date().toISOString().slice(0, 10), runId, 'k27-ab',
          side === 'A' ? commitA : commitB, 'kleinhirn', slug, `L${L}`, 'f16', '', '', '', '', '', '', '', '', '', '', '', '', 'diagnostic',
          `bench/results/k27-ab-${commitA}-${commitB}.json; load ${loadStart.toFixed(2)}`].join('\t') + '\n');
        const browser = await chromium.launch({ headless: false });
        try {
          const page = await browser.newPage();
          await page.goto(`http://localhost:${port}/bench/k28-latency.html?model=${slug}&L=${L}&n=300`);
          await page.waitForFunction(() => window.khK28LatencyResult?.done, null, { timeout: 20 * 60000 });
          const res = await page.evaluate(() => ({ e: window.khK28LatencyResult.error, m: window.khK28LatencyResult.latency?.medianMs }));
          if (res.e) throw new Error(res.e);
          out.cells[key][side].push({ median: res.m, loadStart });
          console.log(`${key} ${side} rep ${r + 1}: ${res.m.toFixed(3)} ms (load ${loadStart.toFixed(2)})`);
        } finally { await browser.close(); }
      }
    }
    const a = out.cells[key].A.map((x) => x.median); const b = out.cells[key].B.map((x) => x.median);
    out.cells[key].ratio = median(b) / median(a);
    console.log(`${key}: A ${median(a).toFixed(3)} B ${median(b).toFixed(3)} B/A ${out.cells[key].ratio.toFixed(3)}`);
  }
} finally {
  for (const s of servers) { try { process.kill(-s.pid, 'SIGTERM'); } catch { /* gone */ } }
  sh('git', ['-C', OFFICIAL, 'checkout', '-q', '--detach', original]);
  writeFileSync(resolve(K28, `bench/results/k27-ab-${commitA}-${commitB}.json`), JSON.stringify(out, null, 1));
}
