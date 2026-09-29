// K20 diagnostic: where ORT Web places the nodes of a small-upstream graph.
// Opens bench/ort.html with log=verbose (session created, one item run, no
// latency), collects the browser console and extracts ORT's node-placement
// lines. Not a latency run; it still writes one runs.tsv line (protocol).
// Usage: node bench/run-ort-log.mjs <f16|f32> <std|opt|optgc> [webgpu|wasm] [capture]
// Output: bench/results/k20-placement-small-upstream-<prec>-<ep>-<graph>[-capture].json
// capture: enableGraphCapture; a failing session create lands in `error`.

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { appendFileSync, writeFileSync } from 'node:fs';

const [precision = 'f16', graph = 'std', ep = 'webgpu', mode = ''] = process.argv.slice(2);
const capture = mode === 'capture';
const BASE = 'http://localhost:5199';
const name = `small-upstream-${precision}-${ep}-${graph}${capture ? '-capture' : ''}`;
const file = `bench/results/k20-placement-${name}.json`;
const commit = execFileSync(
  'git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
appendFileSync('data/hillclimb/runs.tsv', [
  new Date().toISOString().slice(0, 10), `k20-placement-${name}-${Date.now()}`,
  'k20-placement', commit, 'ort-web', 'small-upstream', 'L128K16', precision,
  '', '', '', '', '', '', '', '', '', '', '', '', 'diagnostic', file,
].join('\t') + '\n');

const PLACED = /(All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)/;
const browser = await chromium.launch({ headless: false });
try {
  const page = await browser.newPage();
  const lines = [];
  page.on('console', (m) => lines.push(m.text()));
  await page.goto(`${BASE}/bench/ort.html?model=small-upstream&precision=${precision}`
    + `&ep=${ep}&graph=${graph}&log=verbose${capture ? '&capture=1' : ''}`);
  await page.waitForFunction(() => window.khOrtResult?.done, null, { timeout: 300000 });
  const result = await page.evaluate(() => window.khOrtResult);
  const placements = {};
  for (const l of lines) {
    const m = l.match(PLACED);
    if (m) placements[m[2]] = (placements[m[2]] ?? 0) + Number(m[3]);
  }
  const relevant = lines.filter(
    (l) => /placed on|not assigned|fallback|CPUExecutionProvider|capture|replay|Memcpy/i.test(l));
  const captureLines = lines.filter((l) => /capture|replay|Memcpy/i.test(l));
  const cpuNodes = [];
  let inCpu = false;
  for (const l of lines) {
    const head = l.match(/placed on \[(\w+)\]/);
    if (head) inCpu = head[1] === 'CPUExecutionProvider';
    else if (inCpu) {
      const m = l.match(/VerifyEachNodeIsAssignedToAnEp\]\s+(\w+) \((.*)\)\s*$/);
      if (m) cpuNodes.push({ name: m[2], op: m[1] });
      else inCpu = false;
    }
  }
  writeFileSync(file, JSON.stringify({
    date: new Date().toISOString().slice(0, 10), commit, precision, ep, graph, capture,
    browser: `chromium-${browser.version()}`,
    stage: result?.stage, error: result?.error ?? null,
    adapterInfo: result?.adapterInfo ?? null,
    placements, cpuNodes, captureLines: captureLines.slice(0, 100), consoleLines: lines.length, relevant: relevant.slice(0, 400),
  }, null, 1));
  console.log(file, JSON.stringify(placements), `${relevant.length} relevant lines`);
} finally {
  await browser.close();
}
