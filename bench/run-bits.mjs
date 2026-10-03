// K28.1 bit runner: visible Playwright Chromium on bench/bits.html, writes
// bench/results/k28-bits-<model>-<precision>-<commit>[-rerun].json (SHA-256 of
// the logit bytes per golden file and batch size) and one runs.tsv line.
// Usage: node bench/run-bits.mjs <model> <precision> [port] [--rerun] [--allow-dirty]
//        node bench/run-bits.mjs --compare <commitA> <commitB> [suffixA] [suffixB]
// --compare prints equal/unequal per (model, precision, file, B) and exits 1 on
// any difference or missing file.

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';

const RUNS = 'data/hillclimb/runs.tsv';
const OUT = 'bench/results';
const MODELS = ['small-upstream', 'base-upstream', 'multi-upstream', 'julia-1'];
const PRECISIONS = ['f32', 'f16'];

const git = (...args) => execFileSync('git', args, { encoding: 'utf8' }).trim();
const fileFor = (model, precision, commit, suffix = '') =>
  `${OUT}/k28-bits-${model}-${precision}-${commit}${suffix}.json`;

function compare(commitA, commitB, suffixA, suffixB) {
  let unequal = 0;
  let equal = 0;
  let missing = 0;
  for (const model of MODELS) {
    for (const precision of PRECISIONS) {
      const fa = fileFor(model, precision, commitA, suffixA);
      const fb = fileFor(model, precision, commitB, suffixB);
      if (!existsSync(fa) || !existsSync(fb)) {
        console.log(`${model} ${precision}: missing ${!existsSync(fa) ? fa : fb}`);
        missing += 1;
        continue;
      }
      const a = JSON.parse(readFileSync(fa, 'utf8'));
      const b = JSON.parse(readFileSync(fb, 'utf8'));
      const gpuSame = a.info?.gpuBytes === b.info?.gpuBytes;
      console.log(`${model} ${precision}: gpuBytes ${a.info?.gpuBytes} vs ${b.info?.gpuBytes} ${gpuSame ? 'equal' : 'UNEQUAL'}`);
      if (!gpuSame) unequal += 1;
      for (const file of new Set([...Object.keys(a.files), ...Object.keys(b.files)])) {
        for (const B of ['B1', 'B4']) {
          const ha = a.files[file]?.[B];
          const hb = b.files[file]?.[B];
          const same = ha && hb && ha.sha === hb.sha;
          let detail = '';
          if (!same && ha && hb) {
            const n = Math.min(ha.caseShas.length, hb.caseShas.length);
            const bad = [];
            for (let i = 0; i < n; i += 1) if (ha.caseShas[i] !== hb.caseShas[i]) bad.push(i);
            detail = ` ${bad.length} of ${n} cases differ, first ${bad.slice(0, 5).join(',')}`;
          }
          console.log(`  ${file} ${B}: ${same ? 'equal' : 'UNEQUAL' + detail}`);
          if (same) equal += 1; else unequal += 1;
        }
      }
    }
  }
  console.log(`equal ${equal}, unequal ${unequal}, missing files ${missing}`);
  process.exit(unequal > 0 || missing > 0 ? 1 : 0);
}

const args = process.argv.slice(2);
if (args[0] === '--compare') {
  compare(args[1], args[2], args[3] ?? '', args[4] ?? '');
}

const flags = args.filter((a) => a.startsWith('--'));
const pos = args.filter((a) => !a.startsWith('--'));
const model = pos[0];
const precision = pos[1];
const port = pos[2] ?? '5199';
const rerun = flags.includes('--rerun');
if (!MODELS.includes(model) || !PRECISIONS.includes(precision)) {
  console.error('usage: run-bits.mjs <model> <f32|f16> [port] [--rerun]');
  process.exit(2);
}
const BASE = `http://localhost:${port}`;
const commit = git('rev-parse', '--short', 'HEAD');
const dirty = git('status', '--porcelain', '--', 'src', 'bench/bits.ts', 'bench/bits.html');
if (dirty && !flags.includes('--allow-dirty')) {
  console.error(`engine or runner files are uncommitted, commit first:\n${dirty}`);
  process.exit(2);
}

const COLS = [
  'date', 'run_id', 'change', 'commit', 'engine', 'model', 'bucket', 'precision',
  'argmax_agreement', 'max_abs_logit_diff', 'max_abs_prob_diff', 'median_ms',
  'p95_ms', 'model_only_median_ms', 'load_ms', 'download_mb', 'gpu_mb',
  'peak_mem_mb', 'browser', 'adapter', 'kept', 'note',
];

function appendRun(runId) {
  if (!existsSync(RUNS)) throw new Error(`${RUNS} missing`);
  const cells = COLS.map(() => '');
  Object.assign(cells, {
    0: new Date().toISOString().slice(0, 10), 1: runId, 2: 'k28-bits', 3: commit,
    4: 'kleinhirn', 5: model, 6: 'goldens', 7: precision, 21: 'pending',
  });
  appendFileSync(RUNS, cells.join('\t') + '\n');
}

function finishRun(runId, fields) {
  const lines = readFileSync(RUNS, 'utf8').split('\n');
  const idx = lines.findIndex((l) => l.split('\t')[1] === runId);
  const cells = lines[idx].split('\t');
  for (const [k, v] of Object.entries(fields)) cells[COLS.indexOf(k)] = String(v);
  lines[idx] = cells.join('\t');
  writeFileSync(RUNS, lines.join('\n'));
}

async function main() {
  const suffix = rerun ? '-rerun' : '';
  const runId = `k28-bits-${model}-${precision}-${commit}${suffix}-${Date.now()}`;
  appendRun(runId);
  mkdirSync(OUT, { recursive: true });
  const loadStart = loadavg()[0].toFixed(2);
  const browser = await chromium.launch({ headless: false, args: [] });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.log('[page]', m.text());
    });
    await page.goto(`${BASE}/bench/bits.html?model=${model}&precision=${precision}`);
    const t0 = Date.now();
    let last = '';
    while (Date.now() - t0 < 3600000) {
      const s = await page.evaluate(() => ({
        done: window.khBitsResult?.done ?? false,
        stage: window.khBitsResult?.stage ?? '',
      }));
      if (s.stage !== last && s.stage.endsWith('0')) { last = s.stage; console.log(s.stage); }
      if (s.done) break;
      await new Promise((r) => setTimeout(r, 1000));
    }
    const result = await page.evaluate(() => window.khBitsResult);
    if (!result?.done) throw new Error('timeout');
    if (result.error) {
      finishRun(runId, { kept: 'error', note: result.error });
      throw new Error('page: ' + result.error);
    }
    const out = {
      date: new Date().toISOString().slice(0, 10), run_id: runId, commit, model, precision,
      browser: `chromium-${browser.version()}`,
      loadAvgStart: loadStart, loadAvgEnd: loadavg().map((v) => v.toFixed(2)).join(' '),
      ...result,
    };
    delete out.done;
    const file = fileFor(model, precision, commit, suffix);
    writeFileSync(file, JSON.stringify(out, null, 1));
    const sets = Object.values(result.files).flatMap((f) => [f.B1, f.B4]);
    const agree = Math.min(...sets.map((s) => s.argmaxAgreement ?? 1));
    const diff = Math.max(...sets.map((s) => s.maxAbsLogitDiff ?? 0));
    finishRun(runId, {
      argmax_agreement: agree.toFixed(4),
      max_abs_logit_diff: diff.toExponential(2),
      gpu_mb: ((result.info?.gpuBytes ?? 0) / 1048576).toFixed(1),
      browser: out.browser,
      adapter: JSON.stringify(result.adapterInfo ?? {}).replace(/\t/g, ' '),
      kept: '',
      note: `${file}; load ${loadStart} -> ${loadavg()[0].toFixed(2)}`,
    });
    console.log(file);
    for (const [name, f] of Object.entries(result.files)) {
      console.log(`${name} cases ${f.cases} B1 ${f.B1.sha.slice(0, 12)} B4 ${f.B4.sha.slice(0, 12)} argmax ${f.B1.argmaxAgreement}/${f.B4.argmaxAgreement}`);
    }
  } finally {
    await browser.close();
  }
}

await main();
