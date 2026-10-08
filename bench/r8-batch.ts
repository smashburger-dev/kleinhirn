// R8 hc9 (docs/R8_WORKORDER.md): throughput of runIdsBatch on the threaded WASM path with batch
// plans against rows one by one on the bucket plan, in one page, order on off off on per model.
// Rows: the golden items of the model (real lengths), all of them in one runIdsBatch call (chunks
// of 16 sorted by length). Result: ms per item per setting and the bit check of the two outputs.
// Query: model=<slug>  task=<task>  threads=8  reps=3  [khpost=<url>]

// @ts-expect-error runtime bundle built by vite lib mode has no d.ts
import { EncoderModel as BundleEncoderModel } from '../dist/kleinhirn.js';
import type { EncoderModel as EncoderModelType } from '../src/encoder.ts';
import { isolation } from './k28/k28-8-common.ts';
import { startPosting } from './kbench/post.ts';

const EncoderModel = BundleEncoderModel as typeof EncoderModelType;

interface Result { stage: string; done?: boolean; error?: string; [k: string]: unknown }
declare global { interface Window { khR8Batch?: Result } }

async function main(): Promise<void> {
  const p = new URLSearchParams(location.search);
  const slug = p.get('model') ?? '';
  const task = p.get('task') ?? '';
  const threads = Number(p.get('threads') ?? '8');
  const reps = Number(p.get('reps') ?? '3');
  const result: Result = { stage: 'boot', model: slug, threads, ua: navigator.userAgent };
  window.khR8Batch = result;
  startPosting(() => window.khR8Batch);
  try {
    Object.assign(result, isolation());
    const golden = await (await fetch(`/tests/golden/k28/${slug}/${task}.json`)).json() as { items: { input_ids: number[]; token_type_ids: number[] }[] };
    const inputs = golden.items.map((it) => ({ inputIds: it.input_ids, typeIds: it.token_type_ids }));
    const runs: Record<string, number[]> = { on: [], off: [] };
    const outs: Record<string, Float32Array[]> = {};
    for (const setting of ['on', 'off', 'off', 'on']) {
      result.stage = `loading ${setting}`;
      const enc = await EncoderModel.load({
        manifestUrl: `/models/k28/${slug}/f32/manifest.json`, precision: 'f32', buckets: [128, 512], limits: 'minimum',
        backend: 'wasm', threads, wasmBatchPlans: setting === 'on',
      });
      result.info = enc.info();
      await enc.runIdsBatch(inputs.slice(0, 32)); // warm-up: plans built, code tiered
      result.stage = `running ${setting}`;
      for (let r = 0; r < reps; r += 1) {
        const t0 = performance.now();
        const o = await enc.runIdsBatch(inputs);
        runs[setting].push((performance.now() - t0) / inputs.length);
        outs[setting] ??= o.map((x) => x.data);
      }
      enc.dispose();
    }
    const med = (xs: number[]): number => [...xs].sort((a, b) => a - b)[xs.length >> 1];
    let equal = true;
    outs.on.forEach((a, i) => { const b = outs.off[i]; if (a.length !== b.length || a.some((v, k) => !Object.is(v, b[k]))) equal = false; });
    Object.assign(result, { items: inputs.length, msPerItem: runs, medianOn: med(runs.on), medianOff: med(runs.off), bitEqual: equal });
    result.stage = 'done';
    result.done = true;
  } catch (e) {
    result.stage = 'error';
    result.error = String(e);
    result.done = true;
  }
}

void main();
