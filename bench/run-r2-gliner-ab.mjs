// R2 stage 4 (docs/R2_WORKORDER.md): GLiNER small-upstream L128 on one thread, the WASM path of an
// older build in dist-ref/ (src/wasm.ts with deberta.wasm) against the plan executor of dist/, ABAB
// in one session (order ref new new ref, three or more rounds with --rounds). Page
// bench/kleinhirn.html with backend=wasm (loadEngine), 1000 corpus texts, "nur Modell" median and
// the K5 parity of every run (argmax 100 %, logit deviation at most 1e-3). One runs.tsv line per
// run (change r2-gliner-ab), result bench/results/<tag>-gliner-ab.json.
// Usage: KBENCH_TAG=r2 node bench/run-r2-gliner-ab.mjs [--rounds 1]

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  PauseError, RESULTS, ROOT, appendRun, buildIdOnDisk, checkEnvironment, finishRun, median, openBrowser, sleep,
  startVite, stopVite,
} from './kbench/runner-lib.mjs';

const TAG = process.env.KBENCH_TAG ?? 'r2';
const args = process.argv.slice(2);
const rounds = Number(args.includes('--rounds') ? args[args.indexOf('--rounds') + 1] : 1);
const commit = execFileSync('git', ['-C', ROOT, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
const refBuild = readFileSync(join(ROOT, 'dist-ref/kleinhirn.js'), 'utf8').match(/buildId\s*:\s*["']([a-z0-9]+)["']/)?.[1] ?? null;
const ctx = { ownPids: new Set([process.pid, process.ppid]), vite: null, base: null };
const file = join(RESULTS, `${TAG}-gliner-ab.json`);
const doc = existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : { runs: [] };

async function run(side, i) {
  const env = await checkEnvironment(ctx);
  const runId = `${TAG}-gliner-ab-${side}-${i}-${Date.now()}`;
  appendRun(runId, { change: 'r2-gliner-ab', commit, engine: side === 'ref' ? 'kleinhirn-ref' : 'kleinhirn', model: 'small-upstream',
    bucket: 'L128', precision: 'wasm-f32', note: `${side === 'ref' ? `dist-ref ${refBuild} (deberta.wasm)` : `dist ${buildIdOnDisk()} (plan.wasm)`}, one thread` });
  const drv = await openBrowser('chromium', 'khResult');
  try {
    await drv.goto(`${ctx.base}/bench/kleinhirn.html?model=small-upstream&precision=f32&backend=wasm&goldens=texts1000_l128k16&buckets=128${side === 'ref' ? '&bundle=ref' : ''}`);
    const t0 = Date.now();
    let st;
    while (Date.now() - t0 < 30 * 60000) {
      st = await drv.state();
      if (st.done) break;
      await sleep(1000);
    }
    const r = await drv.result();
    if (!st?.done || r?.error) throw new Error(`${side}: ${r?.error ?? `timeout at ${st?.stage}`}`);
    const rec = { runId, side, buildId: r.info?.buildId ?? null, backend: r.info?.backend ?? null, loadStart: env.loadStart,
      modelOnlyMs: r.modelOnly.medianMs, modelOnlyP95: r.modelOnly.p95Ms, endToEndMs: r.endToEnd.medianMs, n: r.n,
      argmax: r.parity.argmaxAgreement, maxAbsLogitDiff: r.parity.maxAbsLogitDiff };
    rec.parity = rec.argmax === 1 && rec.maxAbsLogitDiff <= 1e-3;
    finishRun(runId, { argmax_agreement: rec.argmax.toFixed(4), max_abs_logit_diff: rec.maxAbsLogitDiff.toExponential(2),
      median_ms: rec.endToEndMs.toFixed(3), model_only_median_ms: rec.modelOnlyMs.toFixed(3), p95_ms: rec.modelOnlyP95.toFixed(3),
      browser: drv.label, kept: '', note: `${side} build ${rec.buildId}; load ${env.loadStart}` });
    console.log(`${runId}: model-only ${rec.modelOnlyMs.toFixed(3)} ms, e2e ${rec.endToEndMs.toFixed(3)}, argmax ${rec.argmax}, logit ${rec.maxAbsLogitDiff.toExponential(2)} (load ${env.loadStart})`);
    return rec;
  } finally {
    await drv.close();
  }
}

try {
  await startVite(ctx);
  for (let k = 0; k < rounds; k += 1) {
    for (const [j, side] of ['ref', 'new', 'new', 'ref'].entries()) {
      doc.runs.push(await run(side, doc.runs.length + j));
      writeFileSync(file, JSON.stringify(doc, null, 1));
    }
  }
} catch (e) {
  if (e instanceof PauseError) { console.error(`PAUSE: ${e.message}`); process.exitCode = 3; } else throw e;
} finally {
  stopVite(ctx);
}
const ms = (side) => doc.runs.filter((r) => r.side === side).map((r) => r.modelOnlyMs);
if (ms('ref').length && ms('new').length) {
  Object.assign(doc, { commit, refBuild, newBuild: buildIdOnDisk(), refMs: median(ms('ref')), newMs: median(ms('new')),
    parity: doc.runs.every((r) => r.parity) });
  doc.change = doc.newMs / doc.refMs - 1;
  writeFileSync(file, JSON.stringify(doc, null, 1));
  console.log(`gliner ab: ref ${doc.refMs.toFixed(3)} new ${doc.newMs.toFixed(3)} ms (${(100 * doc.change).toFixed(2)} %), parity ${doc.parity}`);
}
