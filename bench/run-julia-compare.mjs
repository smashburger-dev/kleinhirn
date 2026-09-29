// K8-6 comparison runner: their benchmark-webgpu.html unchanged (batch 4),
// their library batch 1 (bench-batch1.html), kleinhirn batch 1
// (bench/julia-bench.html), each in its own Playwright Chromium with
// footprint sampling. Load average goes into the JSON; over 4 the whole
// file is marked preliminary.
// Usage: node bench/run-julia-compare.mjs [precision]

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import { mkdirSync, readFileSync, appendFileSync, writeFileSync } from 'node:fs';
import { median, METRICS, ownTreePids } from './mem.mjs';

const ORT = 'http://localhost:5198';
const KH = 'http://localhost:5199';
const RUNS = 'data/hillclimb/runs.tsv';
const MB = 1024 * 1024;
const precision = process.argv[2] ?? 'f16';

function gitCommit() {
  return execFileSync('git', ['rev-parse', '--short', 'HEAD'],
    { encoding: 'utf8' }).trim();
}

function loadAvg() {
  return loadavg().map((v) => v.toFixed(2)).join(' ');
}

function appendRun(runId, engine) {
  const line = [
    new Date().toISOString().slice(0, 10), runId, 'k8-compare', gitCommit(),
    engine, 'julia-1', 'L1024K20', precision,
    '', '', '', '', '', '', '', '', '', '', '', '', '', 'pending',
  ].join('\t');
  appendFileSync(RUNS, line + '\n');
}

function finishRun(runId, fields) {
  const lines = readFileSync(RUNS, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  const cells = lines[idx].split('\t');
  const cols = [
    'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket', 'precision',
    'argmax_agreement', 'max_abs_logit_diff', 'max_abs_prob_diff', 'median_ms',
    'p95_ms', 'model_only_median_ms', 'load_ms', 'download_mb', 'gpu_mb',
    'peak_mem_mb', 'browser', 'adapter', 'kept', 'note',
  ];
  for (const [k, v] of Object.entries(fields)) cells[cols.indexOf(k)] = String(v);
  lines[idx] = cells.join('\t');
  writeFileSync(RUNS, lines.join('\n'));
}

// Runs one phase: opens the page, optionally clicks #run, polls the given
// getter until a JSON status appears, samples footprint meanwhile.
async function phase(cfg) {
  const browser = await chromium.launch({ headless: false, args: [] });
  const samples = [];
  let timer = null;
  try {
    const page = await browser.newPage();
    await new Promise((r) => setTimeout(r, 1500));
    const metricFn = METRICS[cfg.metric];
    const tick = async () => {
      try {
        const pids = ownTreePids();
        samples.push({ t: Date.now(), bytes: metricFn(pids) });
      } catch { /* tree mid-change */ }
    };
    timer = setInterval(tick, 100);
    const load = loadAvg();
    await page.goto(cfg.url);
    if (cfg.button) await page.click(cfg.button);
    const t0 = Date.now();
    let text = '';
    while (Date.now() - t0 < 1200000) {
      text = await page.evaluate(cfg.getter);
      if (cfg.done(text)) break;
      await new Promise((r) => setTimeout(r, 1000));
    }
    let downloadBytes = null;
    if (cfg.resources) {
      downloadBytes = await page.evaluate(() =>
        performance.getEntriesByType('resource')
          .reduce((n, e) => n + (e.transferSize || e.encodedBodySize || 0), 0));
    }
    await tick();
    clearInterval(timer);
    const baselineMb = median(samples.slice(0, 10).map((s) => s.bytes)) / MB;
    const peakMb = Math.max(...samples.map((s) => s.bytes)) / MB;
    return {
      result: text,
      loadAvg: load,
      baselineMb,
      peakMemMb: peakMb - baselineMb,
      downloadBytes,
      browser: `chromium-${browser.version()}`,
      samples: samples.length,
    };
  } finally {
    if (timer) clearInterval(timer);
    await browser.close();
  }
}

const phases = [
  {
    name: 'ort-webgpu-batch4',
    engine: 'onnxruntime-web',
    url: `${ORT}/benchmark-webgpu.html`,
    button: '#run',
    resources: true,
    getter: () => document.querySelector('#status').textContent,
    done: (t) => t.startsWith('{') || t.includes('rror'),
  },
  {
    name: 'ort-webgpu-batch1',
    engine: 'onnxruntime-web',
    url: `${ORT}/bench-batch1.html`,
    button: '#run',
    resources: true,
    getter: () => window.juliaBatch1Result
      ?? document.querySelector('#status').textContent,
    done: (t) => t.startsWith('{') || t.includes('rror'),
  },
  {
    name: `kleinhirn-${precision}-batch1`,
    engine: 'kleinhirn',
    url: `${KH}/bench/julia-bench.html?precision=${precision}&limits=minimum`,
    button: null,
    getter: () => (window.khJuliaBench?.done ? JSON.stringify(
      { result: window.khJuliaBench.result ?? null,
        info: window.khJuliaBench.info ?? null,
        error: window.khJuliaBench.error ?? null }) : ''),
    done: (t) => t.length > 0,
  },
];

const out = {
  date: new Date().toISOString().slice(0, 10),
  commit: gitCommit(),
  metric: 'footprint',
  runs: {},
};
// Calibrated metric name, same source as run-kleinhirn.mjs.
try {
  out.metric = JSON.parse(readFileSync(
    'bench/results/calibration-latest.json', 'utf8')).chosen ?? 'footprint';
} catch { /* default footprint */ }
mkdirSync('bench/results', { recursive: true });
for (const cfg of phases) {
  const runId = `${cfg.name}-${Date.now()}`;
  appendRun(runId, cfg.engine);
  const r = await phase({ ...cfg, metric: out.metric });
  let parsed = null;
  try { parsed = JSON.parse(r.result); } catch { /* status not JSON */ }
  if (parsed?.error) {
    out.runs[cfg.name] = { error: parsed.error, loadAvg: r.loadAvg };
  } else {
    const res = parsed?.result ?? parsed;
    out.runs[cfg.name] = {
      loadAvg: r.loadAvg,
      browser: r.browser,
      baselineMb: r.baselineMb,
      peakMemMb: r.peakMemMb,
      ...res,
      download_bytes: parsed?.info?.downloadBytes ?? r.downloadBytes,
      gpuBytes: parsed?.info?.gpuBytes,
      loadTiming: parsed?.info?.loadTiming,
    };
    const per = res?.ms_per_request ?? res?.per_request_median_ms;
    const dl = parsed?.info?.downloadBytes ?? r.downloadBytes;
    finishRun(runId, {
      argmax_agreement: res?.matching_predictions != null
        ? (res.matching_predictions / (res.requests || 100)).toFixed(4) : '',
      max_abs_logit_diff: res?.max_abs_logit_error_vs_pytorch != null
        ? res.max_abs_logit_error_vs_pytorch.toExponential(2) : '',
      median_ms: per != null ? per.toFixed(2) : '',
      load_ms: res?.load_and_warm_ms != null
        ? res.load_and_warm_ms.toFixed(0) : (res?.load_ms?.toFixed(0) ?? ''),
      download_mb: dl != null ? (dl / MB).toFixed(1) : '',
      gpu_mb: parsed?.info?.gpuBytes != null
        ? (parsed.info.gpuBytes / MB).toFixed(0) : '',
      peak_mem_mb: r.peakMemMb.toFixed(0),
      browser: r.browser,
      kept: '',
      note: `loadAvg ${r.loadAvg}`,
    });
  }
}
const oneMinLoad = (s) => Number(String(s ?? '').match(/[\d.]+/)?.[0] ?? 0);
const worstLoad = Math.max(...Object.values(out.runs)
  .map((r) => oneMinLoad(r.loadAvg)));
out.preliminary = worstLoad > 4;
out.worstLoadAvg = worstLoad;
const file = `bench/results/${out.date}-julia-compare.json`;
writeFileSync(file, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
console.log('wrote', file);
