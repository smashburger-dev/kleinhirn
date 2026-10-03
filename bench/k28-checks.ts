// K28.8 phase 1 step 8: cost of the K28.R checks per engine call (FINDINGS §39,
// open point), without a change under src/. On one device:
// - scope: 1.000 calls of push/popErrorScope('validation') around a submit of an
//   empty command buffer, awaited together with onSubmittedWorkDone, against
//   1.000 calls of the same submit without the scope (alternating blocks of 100);
// - finite: assertFinite (src/tasks.ts) on 384 and on 512 x 3 values, 10.000 calls each.
// Result in µs per call. Query: none.

import { assertFinite } from '../src/tasks.ts';
import { median } from './metrics.ts';
import { isolation } from './k28/k28-8-common.ts';

interface Result {
  stage: string; crossOriginIsolated?: boolean; timerStepMs?: number; adapterInfo?: unknown;
  scope?: Record<string, number>; finite?: Record<string, number>; error?: string; done?: boolean;
}
declare global {
  interface Window { khK28ChecksResult?: Result }
}

const sorted = (xs: number[]): number[] => [...xs].sort((a, b) => a - b);

async function main(): Promise<void> {
  const result: Result = { stage: 'boot' };
  window.khK28ChecksResult = result;
  try {
    Object.assign(result, isolation());
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no adapter');
    result.adapterInfo = { vendor: adapter.info.vendor, architecture: adapter.info.architecture };
    const device = await adapter.requestDevice();
    const submitEmpty = (): void => device.queue.submit([device.createCommandEncoder().finish()]);
    const plain = async (): Promise<void> => {
      submitEmpty();
      await device.queue.onSubmittedWorkDone();
    };
    const scoped = async (): Promise<void> => {
      device.pushErrorScope('validation');
      submitEmpty();
      const popped = device.popErrorScope();
      const [err] = await Promise.all([popped, device.queue.onSubmittedWorkDone()]);
      if (err) throw new Error(err.message);
    };
    for (let i = 0; i < 100; i += 1) { await plain(); await scoped(); }
    result.stage = 'scope';
    const a: number[] = []; const b: number[] = [];
    for (let block = 0; block < 20; block += 1) {
      const [fn, out] = block % 2 === 0 ? [plain, a] : [scoped, b];
      for (let i = 0; i < 100; i += 1) {
        const t = performance.now();
        await fn();
        out.push(performance.now() - t);
      }
    }
    // 10 blocks of 100 each: 1.000 calls per variant.
    const ma = median(sorted(a)) * 1000; const mb = median(sorted(b)) * 1000;
    const meanA = (a.reduce((s, x) => s + x, 0) / a.length) * 1000;
    const meanB = (b.reduce((s, x) => s + x, 0) / b.length) * 1000;
    result.scope = {
      calls: b.length, plainMedianUs: ma, scopedMedianUs: mb, deltaMedianUs: mb - ma,
      plainMeanUs: meanA, scopedMeanUs: meanB, deltaMeanUs: meanB - meanA,
    };

    result.stage = 'finite';
    const finite: Record<string, number> = {};
    for (const [name, len] of [['n384', 384], ['n1536', 512 * 3]] as const) {
      const v = new Float32Array(len).map((_, i) => Math.sin(i));
      for (let i = 0; i < 1000; i += 1) assertFinite(v, 'warm');
      const t = performance.now();
      for (let i = 0; i < 10000; i += 1) assertFinite(v, 'bench');
      finite[`${name}UsPerCall`] = ((performance.now() - t) / 10000) * 1000;
    }
    result.finite = finite;
    device.destroy();
    result.stage = 'done';
    result.done = true;
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
  }
}

void main();
