// K28.4 parity runner: visible Playwright Chromium on bench/k28-parity.html,
// writes bench/results/k28-parity-<slug>-<precision>-<commit>[-layers].json and
// one runs.tsv line (change k28.4-parity).
// --text (K28.3) runs the text methods against the same goldens (G2, G3); change
// k28.3-text, results in ...-text.json.
// Usage: node bench/run-k28-parity.mjs <slug> <f32|f16> [port] [--layers|--text] [--allow-dirty]
//        node bench/run-k28-parity.mjs --pilot|--pilot-k28.5 <f32|f16> [port]
// --pilot runs the eight pilot models one after the other; --pilot-k28.5 (K28.5) runs
// the sixteen RoBERTa, XLM-R and DistilBERT models, change k28.5-parity; --pilot-k28.6
// (K28.6) the eleven DeBERTa and ModernBERT models, change k28.6-parity. The four XLM-R
// models regenerated in K28.6 step 0 also run under k28.6-parity (--k28.6 on a single slug).
// --r6 (R6 G6) runs the six bench models, change r6-g6-parity, results in ...-r6g6.json with
// the SHA-256 of every output; tools/r6_g6.mjs compares two commits. --bundle loads dist/kleinhirn.js
// (results ...-r6g6b.json, with the loaded script URLs), --wasm the WASM executor (...-r6g6bw.json).

import { chromium } from '@playwright/test';
import { execFileSync, spawnSync } from 'node:child_process';
import { loadavg } from 'node:os';
import {
  appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync,
} from 'node:fs';

const RUNS = 'data/hillclimb/runs.tsv';
const OUT = 'bench/results';
const PILOT = [
  'daekeun-ml__koelectra-small-v3-nsmc',
  'MoritzLaurer__xtremedistil-l6-h256-zeroshot-v1.1-all-33',
  'sentence-transformers__all-MiniLM-L6-v2',
  'BAAI__bge-small-en-v1.5',
  'cross-encoder__ms-marco-MiniLM-L4-v2',
  'cross-encoder__ms-marco-MiniLM-L6-v2',
  'cross-encoder__ms-marco-MiniLM-L12-v2',
  'dslim__bert-base-NER',
];
const PILOT_K285 = [
  'cardiffnlp__twitter-roberta-base-sentiment-latest',
  'cross-encoder__nli-distilroberta-base',
  'sentence-transformers__all-distilroberta-v1',
  'OpenMed__OpenMed-NER-OrganismDetect-TinyMed-82M',
  'cross-encoder__stsb-distilroberta-base',
  'qilowoq__mmarco-mMiniLMv2-L12-H384-v1-en-ru',
  'MoritzLaurer__multilingual-MiniLMv2-L6-mnli-xnli',
  'd0rj__e5-small-en-ru',
  'ukr-models__uk-ner',
  'cross-encoder__mmarco-mMiniLMv2-L12-H384-v1',
  'distilbert__distilbert-base-uncased-finetuned-sst-2-english',
  'typeform__distilbert-base-uncased-mnli',
  'sentence-transformers__distiluse-base-multilingual-cased-v1',
  'OpenMed__OpenMed-NER-BloodCancerDetect-TinyMed-65M',
  'Amdestya__ce-cat-distilbert',
  'emrecan__distilbert-base-turkish-cased-allnli_tr',
];
const PILOT_K286 = [
  'protectai__deberta-v3-base-prompt-injection-v2',
  'cross-encoder__nli-deberta-v3-small',
  'xushijie__polyBERT',
  'OpenMed__OpenMed-NER-ProteinDetect-SuperClinical-141M',
  'mixedbread-ai__mxbai-rerank-xsmall-v1',
  'sheltron-ai__prompt-guard-68m',
  'Horizon-Labs__multilingual-zeroshot-small',
  'ibm-granite__granite-embedding-small-english-r2',
  'OpenMed__OpenMed-NER-ChemicalDetect-ModernMed-149M',
  'hotchpotch__japanese-reranker-xsmall-v2',
  'ibm-granite__granite-embedding-reranker-english-r2',
];
// R6 G6: the six bench models of the kbench table (bench/kbench/runner-lib.mjs MODELS).
const R6 = [
  'sentence-transformers__all-MiniLM-L6-v2',
  'cardiffnlp__twitter-roberta-base-sentiment-latest',
  'cross-encoder__mmarco-mMiniLMv2-L12-H384-v1',
  'distilbert__distilbert-base-uncased-finetuned-sst-2-english',
  'protectai__deberta-v3-base-prompt-injection-v2',
  'ibm-granite__granite-embedding-small-english-r2',
];
// The pilot groups: flag, slugs, change prefix.
const GROUPS = [
  ['--r6', R6, 'r6-g6'],
  ['--pilot', PILOT, 'k28.4'],
  ['--pilot-k28.5', PILOT_K285, 'k28.5'],
  ['--pilot-k28.6', PILOT_K286, 'k28.6'],
];
const PRECISIONS = ['f32', 'f16'];

const git = (...args) => execFileSync('git', args, { encoding: 'utf8' }).trim();
const args = process.argv.slice(2);
const flags = args.filter((a) => a.startsWith('--'));
// R4: --words rows|progressive takes its value as the next argument, not as a positional one
const wordsAt = args.indexOf('--words');
const wordsVal = wordsAt >= 0 ? args[wordsAt + 1] : undefined;
const words = wordsVal ?? 'full';
if (wordsAt >= 0 && !['rows', 'progressive'].includes(wordsVal)) {
  console.error('usage: --words rows|progressive');
  process.exit(2);
}
const pos = args.filter((a) => !a.startsWith('--') && a !== wordsVal);

const group = GROUPS.find(([flag]) => flags.includes(flag));
if (group) {
  const [precision, port = '5199'] = pos;
  if (!PRECISIONS.includes(precision)) {
    console.error('usage: run-k28-parity.mjs --r6|--pilot|--pilot-k28.5|--pilot-k28.6 <f32|f16> [port]');
    process.exit(2);
  }
  let failed = 0;
  for (const slug of group[1]) {
    const r = spawnSync('node', [
      'bench/run-k28-parity.mjs', slug, precision, port,
      ...flags.filter((f) => f !== '--words' && !GROUPS.some(([flag]) => flag === f)),
      ...(wordsVal ? ['--words', wordsVal] : []),
      ...(group[0] === '--r6' ? ['--r6-g6'] : [])], { stdio: 'inherit' });
    if (r.status !== 0) failed += 1;
  }
  process.exit(failed ? 1 : 0);
}

const [slug, precision, port = '5199'] = pos;
const layers = flags.includes('--layers');
const text = flags.includes('--text');
// R6: --bundle runs dist/kleinhirn.js (build first), --wasm the WASM executor (f32 only)
const bundle = flags.includes('--bundle');
const wasmBackend = flags.includes('--wasm');
const PILOTS = [...PILOT, ...PILOT_K285, ...PILOT_K286];
// change prefix of the run: the K28.6 models and, with --k28.6, the XLM-R models of step 0
// (R6: --r6-g6 marks the G6 bit runs, change r6-g6-parity)
const generation = flags.includes('--r6-g6') ? 'r6-g6' : PILOT_K286.includes(slug) || flags.includes('--k28.6') ? 'k28.6'
  : PILOT_K285.includes(slug) ? 'k28.5' : PILOTS.includes(slug) ? 'k28.4' : 'k28.s';
// K28.S: every model of the list is allowed; its change prefix is k28.s
const LIST = JSON.parse(readFileSync('data/k28/models.json', 'utf8')).models.map((e) => e.id.replace('/', '__'));
if (!LIST.includes(slug) || !PRECISIONS.includes(precision)) {
  console.error('usage: run-k28-parity.mjs <slug> <f32|f16> [port] [--layers]');
  process.exit(2);
}
const BASE = `http://localhost:${port}`;
const commit = git('rev-parse', '--short', 'HEAD');
const dirty = git('status', '--porcelain', '--', 'src', 'bench/k28-parity.ts', 'bench/k28-parity.html');
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
    0: new Date().toISOString().slice(0, 10), 1: runId,
    2: (text && !['k28.6', 'k28.s', 'r6-g6'].includes(generation) ? 'k28.3-text' : `${generation}-parity${layers ? '-layers' : ''}`)
      + (wordsVal ? `-words-${words}` : ''), 3: commit,
    4: 'kleinhirn', 5: slug, 6: '128+512', 7: precision, 21: 'pending',
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

// The argmax-like agreement of the run: the smallest of the task's rates.
function agreementOf(m) {
  const rates = [m.argmaxAgreement, m.zeroShotAgreement, m.bestPassageAgreement,
    m.argmaxPerTokenAgreement, m.spanExactAgreement].filter((v) => typeof v === 'number');
  return rates.length ? Math.min(...rates) : '';
}

async function main() {
  // R4: r6g6 + b (bundle) + w (wasm) + r (rows) or p (progressive)
  const wordsSuffix = words === 'rows' ? 'r' : words === 'progressive' ? 'p' : '';
  const suffix = layers ? '-layers' : text ? '-text' : generation === 'r6-g6' ? `-r6g6${bundle ? 'b' : ''}${wasmBackend ? 'w' : ''}${wordsSuffix}`
    : wordsVal ? `-words-${words}` : '';
  const runId = `k28-parity-${slug}-${precision}-${commit}${suffix}-${Date.now()}`;
  appendRun(runId);
  mkdirSync(OUT, { recursive: true });
  const loadStart = loadavg()[0].toFixed(2);
  const browser = await chromium.launch({ headless: false, args: [] });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.log('[page]', m.text());
    });
    await page.goto(`${BASE}/bench/k28-parity.html?model=${slug}&precision=${precision}${layers ? '&layers=1' : ''}${text ? '&text=1' : ''}${bundle ? '&bundle=1' : ''}${wasmBackend ? '&backend=wasm' : ''}${wordsVal ? `&words=${words}` : ''}`);
    const t0 = Date.now();
    while (Date.now() - t0 < 1800000) {
      const done = await page.evaluate(() => window.khK28ParityResult?.done ?? false);
      if (done) break;
      await new Promise((r) => setTimeout(r, 1000));
    }
    const result = await page.evaluate(() => window.khK28ParityResult);
    if (!result?.done) throw new Error('timeout');
    if (result.error) {
      finishRun(runId, { kept: 'error', note: result.error });
      throw new Error('page: ' + result.error);
    }
    // --bundle: the dev server can serve a stale transform of dist/ (vite.config.ts), so the page's
    // build id must be the one of the file on disk
    if (bundle) {
      const disk = readFileSync('dist/kleinhirn.js', 'utf8').match(/buildId\s*:\s*["']([a-z0-9]+)["']/)?.[1];
      if (!disk || result.info?.buildId !== disk) {
        finishRun(runId, { kept: 'error', note: `stale bundle: page ${result.info?.buildId}, disk ${disk}` });
        throw new Error(`stale bundle: the page ran build ${result.info?.buildId}, dist/kleinhirn.js is ${disk}; restart the dev server`);
      }
    }
    const out = {
      date: new Date().toISOString().slice(0, 10), run_id: runId, commit,
      browser: `chromium-${browser.version()}`,
      loadAvgStart: loadStart, loadAvgEnd: loadavg().map((v) => v.toFixed(2)).join(' '),
      ...result,
    };
    delete out.done;
    const file = `${OUT}/k28-parity-${slug}-${precision}-${commit}${suffix}.json`;
    writeFileSync(file, JSON.stringify(out, null, 1));
    const m = result.metrics ?? {};
    finishRun(runId, {
      argmax_agreement: layers ? '' : (typeof agreementOf(m) === 'number' ? agreementOf(m).toFixed(4) : ''),
      max_abs_logit_diff: typeof (m.maxAbsLogitDiff ?? m.maxAbsDiffFinal ?? m.maxAbsScoreDiff) === 'number'
        ? (m.maxAbsLogitDiff ?? m.maxAbsDiffFinal ?? m.maxAbsScoreDiff).toExponential(2) : '',
      gpu_mb: ((result.info?.gpuBytes ?? 0) / 1048576).toFixed(1),
      browser: out.browser,
      adapter: JSON.stringify(result.adapterInfo ?? {}).replace(/\t/g, ' '),
      kept: '',
      note: `${file}; load ${loadStart} -> ${loadavg()[0].toFixed(2)}${m.textStats
        ? `; text bitDiffRows ${m.textStats.bitDiffRows}/${m.textStats.rows} inputMismatch ${m.textStats.inputMismatch}` : ''}`,
    });
    console.log(file);
    console.log(JSON.stringify({ task: result.task, metrics: { ...result.metrics, firstDifferingText: undefined }, g4: result.g4,
      layers: result.layers && { worst: result.layers.worstLayer,
        max: Math.max(...result.layers.maxAbsDiff), maxAbsDiff: result.layers.maxAbsDiff } }));
  } finally {
    await browser.close();
  }
}

await main();
