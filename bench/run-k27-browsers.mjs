// K27: the K28 parity page (bench/k28-parity.html) in Playwright Firefox and WebKit, to check
// the new kernels (mmtile, matmul attention) outside Chromium. One runs.tsv line per run
// (change k27-browsers), result bench/results/k27-browser-<browser>-<slug>-<precision>-<commit>.json.
// Prints the decision rate and largest deviation next to the Chromium file of the same commit.
// Usage: node bench/run-k27-browsers.mjs <firefox|webkit> <slug>[,<slug>] <f32|f16>[,...] [port]

import { firefox, webkit } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { loadavg } from 'node:os';

const [name, slugs, precisions, port = '5199'] = process.argv.slice(2);
const type = { firefox, webkit }[name];
if (!type) throw new Error('usage: run-k27-browsers.mjs <firefox|webkit> <slugs> <precisions> [port]');
const commit = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
const rate = (m) => m.argmaxAgreement ?? m.bestPassageAgreement ?? m.argmaxPerTokenAgreement ?? m.minCosineFinal;
const diff = (m) => m.maxAbsLogitDiff ?? m.maxAbsDiffFinal;

for (const slug of slugs.split(',')) {
  for (const precision of precisions.split(',')) {
    const runId = `k27-browser-${name}-${slug}-${precision}-${Date.now()}`;
    const file = `bench/results/k27-browser-${name}-${slug}-${precision}-${commit}.json`;
    appendFileSync('data/hillclimb/runs.tsv', [new Date().toISOString().slice(0, 10), runId, 'k27-browsers', commit,
      'kleinhirn', slug, '128+512', precision, '', '', '', '', '', '', '', '', '', '', name, '', 'diagnostic',
      `${file}; load ${loadavg()[0].toFixed(2)}`].join('\t') + '\n');
    const browser = await type.launch({ headless: false });
    let result;
    try {
      const page = await browser.newPage();
      await page.goto(`http://localhost:${port}/bench/k28-parity.html?model=${slug}&precision=${precision}`);
      await page.waitForFunction(() => window.khK28ParityResult?.done, null, { timeout: 30 * 60000 });
      result = await page.evaluate(() => window.khK28ParityResult);
      writeFileSync(file, JSON.stringify({ run_id: runId, commit, browser: `${name}-${browser.version()}`, ...result }, null, 1));
    } finally {
      await browser.close();
    }
    const chromiumFile = `bench/results/k28-parity-${slug}-${precision}-${process.env.REF_COMMIT ?? commit}.json`; // REF_COMMIT: the commit of the Chromium parity file
    const ref = existsSync(chromiumFile) ? JSON.parse(readFileSync(chromiumFile, 'utf8')).metrics : null;
    const m = result.metrics ?? {};
    console.log(`${name} ${slug} ${precision}: ${result.error ? `ERROR ${result.error.slice(0, 200)}`
      : `rate ${rate(m)} maxdiff ${diff(m)?.toExponential(2)} g4 ${result.g4?.differingRows}/${result.g4?.rows} gpuErrors ${result.gpuErrors?.length}`}`
      + (ref ? ` | chromium rate ${rate(ref)} maxdiff ${diff(ref)?.toExponential(2)}` : ' | no chromium file at this commit'));
  }
}
