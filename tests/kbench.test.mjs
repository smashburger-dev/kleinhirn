// R1 unit tests: cell matrix, bucket rule, block measurement and table summary.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { bucketFor as bucketForTs } from '../bench/kbench/engine.ts';
import { blockForTimerStep as blockTs, measureWindow } from '../bench/kbench/measure.ts';
import {
  MODELS, bucketFor, blockForTimerStep, cellMatrix, compareToReference, engineOrder, missingReason,
  parityVerdict, renderMarkdown, summarizeTable, parseFilters, BROWSERS, WAYS, LENGTHS, geoChange,
} from '../bench/kbench/runner-lib.mjs';

const all = { browsers: BROWSERS, ways: WAYS, models: MODELS.map((m) => m.slug), lengths: LENGTHS };

test('matrix: 6 models x 3 lengths x 3 ways x 4 browsers = 216 cells per engine', () => {
  const cells = cellMatrix(all);
  assert.equal(cells.length, 216);
  assert.equal(new Set(cells.map((c) => `${c.browser}|${c.way}|${c.slug}|${c.len}`)).size, 216);
  // run order: browser, then way, then model and length
  assert.deepEqual(cells[0], { browser: 'chromium', way: 'webgpu-f16', slug: MODELS[0].slug, len: '128' });
  assert.equal(cells[18].way, 'webgpu-f32');
  assert.equal(cells[54].browser, 'webkit');
});

test('matrix: filters', () => {
  const f = parseFilters(['--browsers', 'chromium', '--ways', 'f16,wasm', '--models', 'minilm,granite', '--lengths', 'real']);
  const cells = cellMatrix(f);
  assert.equal(cells.length, 1 * 2 * 2 * 1);
  assert.deepEqual([...new Set(cells.map((c) => c.way))], ['webgpu-f16', 'wasm']);
  assert.throws(() => parseFilters(['--browsers', 'opera']));
});

test('engine order alternates by cell index', () => {
  assert.deepEqual(engineOrder(0), ['kleinhirn', 'ort']);
  assert.deepEqual(engineOrder(1), ['ort', 'kleinhirn']);
  assert.deepEqual(engineOrder(2), ['kleinhirn', 'ort']);
});

test('bucket rule: smallest of {128, 512} that fits, same in lib and engine contract', () => {
  for (const [len, want] of [[1, 128], [127, 128], [128, 128], [129, 512], [259, 512], [512, 512]]) {
    assert.equal(bucketFor(len), want);
    assert.equal(bucketForTs(len), want);
  }
  assert.throws(() => bucketFor(513));
  assert.equal(bucketFor(200, [64, 256, 1024]), 256);
});

test('block rule: timer step above 0.1 ms measures blocks of 10', () => {
  for (const f of [blockForTimerStep, blockTs]) {
    assert.equal(f(0.005), 1);
    assert.equal(f(0.1), 1);
    assert.equal(f(0.1001), 10);
    assert.equal(f(1), 10);
  }
});

test('measureWindow: block means, copies, progress', async () => {
  let t = 0;
  const now = () => t;
  const seen = [];
  const buf = new Float32Array(2);
  const run = async (i) => { t += i + 1; seen.push(i); buf[0] = i; buf[1] = -i; return buf; };
  const w1 = await measureWindow(run, 10, 4, 1, now);
  assert.deepEqual(w1.samplesMs, [11, 12, 13, 14]);
  assert.deepEqual(seen, [10, 11, 12, 13]);
  assert.deepEqual(w1.outputs.map((o) => o[0]), [10, 11, 12, 13]); // copies, not the reused buffer
  t = 0; seen.length = 0;
  const w2 = await measureWindow(run, 0, 20, 10, now);
  assert.equal(w2.samplesMs.length, 2);
  assert.deepEqual(w2.samplesMs, [(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) / 10, (11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20) / 10]);
  assert.equal(w2.outputs.length, 20);
  await assert.rejects(() => measureWindow(run, 0, 15, 10, now));
});

test('missing reasons: R2, no WebGPU, no shader-f16, probe failures', () => {
  const probe = { env: { webgpu: true, adapter: { shaderF16: true } }, capability: { kleinhirn: { 'webgpu-f16': { startable: true } }, ort: { wasm: { startable: false, error: 'boom' } } } };
  assert.equal(missingReason('chromium', 'wasm', 'kleinhirn', probe), null); // R2: the WASM plan executor
  assert.equal(missingReason('chromium', 'webgpu-f16', 'kleinhirn', probe), null);
  assert.match(missingReason('chromium', 'wasm', 'ort', probe), /boom/);
  assert.match(missingReason('safari', 'webgpu-f16', 'ort', null), /nicht geprobt/);
  assert.match(missingReason('firefox', 'webgpu-f16', 'ort', { env: { webgpu: false } }), /kein WebGPU/);
  assert.match(missingReason('webkit', 'webgpu-f16', 'ort', { env: { webgpu: true, adapter: { shaderF16: false } } }), /shader-f16/);
  assert.match(missingReason('safari', 'wasm', 'ort', { automationError: 'x' }), /Automatisierung/);
});

test('parity verdict: f32 gate, f16 bound of three times delta_sim', () => {
  assert.equal(parityVerdict('webgpu-f32', 'sequence-classification', { argmaxAgreement: 1, maxAbsLogitDiff: 1e-5, finite: true }, null, 0.01).pass, true);
  assert.equal(parityVerdict('webgpu-f32', 'sequence-classification', { argmaxAgreement: 0.995, finite: true }, null, 0.01).pass, false);
  assert.equal(parityVerdict('webgpu-f16', 'sequence-classification', { argmaxAgreement: 0.995, maxAbsLogitDiff: 0.029, finite: true }, null, 0.01).pass, true);
  assert.equal(parityVerdict('webgpu-f16', 'sequence-classification', { argmaxAgreement: 1, maxAbsLogitDiff: 0.031, finite: true }, null, 0.01).pass, false);
  assert.equal(parityVerdict('webgpu-f16', 'embeddings', { minCosine: 0.99995, maxAbsDiff: 0.001, finite: true }, { maxAbsDiff: 0.001, minCosine: 0.9999 }, 0.01).pass, true);
  assert.equal(parityVerdict('webgpu-f16', 'embeddings', { minCosine: 0.9998, finite: true }, null, 0.01).pass, false);
  // normalizing head: the absolute bound of the normalized simulation does not apply, the cosine does
  assert.equal(parityVerdict('webgpu-f16', 'embeddings', { minCosine: 0.99999, maxAbsDiff: 0.0016, finite: true }, { maxAbsDiff: 0.0006, minCosine: 0.99999 }, 0.00031, true).pass, true);
  assert.equal(parityVerdict('webgpu-f16', 'embeddings', { minCosine: 0.99999, maxAbsDiff: 0.0016, finite: true }, null, 0.00031, false).pass, false);
  assert.equal(parityVerdict('webgpu-f16', 'reranking', { bestPassageAgreement: 1, maxAbsLogitDiff: 0.01, finite: false }, null, 0.01).pass, false);
});

test('compareToReference: max deviation, argmax and cosine against the CPU rows', () => {
  const ref = new Float32Array([1, 0, 0, 1, 5, 4, 0, 3]); // rows 0..3, width 2
  const got = new Float32Array([5, 4.5, 0, 3.2]); // rows 2 and 3
  const outputs = { rows: 2, width: 2, base64: Buffer.from(got.buffer).toString('base64') };
  const r = compareToReference(outputs, Buffer.from(ref.buffer), 2, 2, 'sequence-classification');
  assert.equal(r.rows, 2);
  assert.ok(Math.abs(r.maxAbsDiff - 0.5) < 1e-6);
  assert.equal(r.argmaxAgreement, 1);
  assert.throws(() => compareToReference(outputs, Buffer.from(ref.buffer), 2, 3, 'embeddings'));
});

test('table summary: missing reasons, factor, memory, parity', () => {
  const cells = cellMatrix({ browsers: ['chromium'], ways: ['webgpu-f16', 'wasm'], models: [MODELS[0].slug], lengths: ['128'] });
  const run = (median) => ({ runId: 'x', medianMs: median, p95Ms: median * 1.2, downloadMb: 10, engineGzipB: 5, parity: { pass: true, reasons: [] }, config: { build: 'webgpu' } });
  const get = (c, e) => (c.way === 'webgpu-f16' ? run(e === 'ort' ? 6 : 3) : (e === 'ort' ? run(900) : undefined));
  const mem = (c, e) => (e === 'ort' && c.way === 'webgpu-f16' ? { peakMemMib: 321, memNote: null } : null);
  const miss = (c, e) => missingReason(c.browser, c.way, e, { env: { webgpu: true, adapter: { shaderF16: true } }, capability: {} });
  const t = summarizeTable(cells, get, mem, miss);
  assert.equal(t.length, 2);
  assert.equal(t[0].factor, 2);
  assert.equal(t[0].ort.peakMemMib, 321);
  assert.equal(t[0].kleinhirn.memNote, 'nicht gemessen');
  assert.equal(t[1].kleinhirn.status, 'fehlt: nicht gelaufen'); // R2: a WASM cell without a run
  assert.equal(t[1].ort.status, 'ok');
  assert.equal(t[1].factor, null);
  assert.equal(t[0].ort.shortSample, false);
  const md = renderMarkdown(t, { date: '2026-10-03', commit: 'abc', gzip: { kleinhirn: 1, ort: 'jsep 2' } });
  assert.match(md, /### chromium, WebGPU f16/);
  assert.match(md, /fehlt: nicht gelaufen/);
  assert.match(md, /iPhone/);
});

test('matrix: firefox-reg only by filter and only for WASM (R8)', () => {
  const f = parseFilters(['--browsers', 'chromium,firefox-reg', '--lengths', '128']);
  const cells = cellMatrix(f);
  assert.equal(cells.filter((c) => c.browser === 'firefox-reg').length, 6);
  assert.ok(cells.filter((c) => c.browser === 'firefox-reg').every((c) => c.way === 'wasm'));
  assert.equal(cells.filter((c) => c.browser === 'chromium').length, 18);
});

test('geoChange: geometric mean of new / ref minus 1', () => {
  assert.ok(Math.abs(geoChange([{ newMs: 2, refMs: 1 }, { newMs: 1, refMs: 2 }])) < 1e-12);
  assert.ok(Math.abs(geoChange([{ newMs: 98, refMs: 100 }, { newMs: 49, refMs: 50 }]) + 0.02) < 1e-12);
});
