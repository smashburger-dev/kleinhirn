// K27 diagnosis: kleinhirn per-dispatch GPU profile of one K28.8 model at full length L.
// EncoderModel (src) with buckets [L], f16; the PlanExecutor of the bucket times every
// dispatch in its own pass (profileForward 'dispatch'). 20 warmup calls, 20 profiles,
// median per dispatch name; wall time of 50 runIds calls next to it.
// Query: ?model=<slug>&L=128&limits=minimum|default

import { EncoderModel } from '../src/index.ts';
import { median } from './metrics.ts';
import { loadInputs } from './k28/k28-8-common.ts';

interface Result {
  stage: string; info?: Record<string, unknown>; deviceLimits?: Record<string, number>;
  wallMedianMs?: number; gpuSumMedianMs?: number;
  ops?: { name: string; medianUs: number }[]; error?: string; done?: boolean;
}
declare global {
  interface Window { khK28KhProfile?: Result }
}

async function main(): Promise<void> {
  const result: Result = { stage: 'boot' };
  window.khK28KhProfile = result;
  try {
    const p = new URLSearchParams(location.search);
    const slug = p.get('model') ?? '';
    const L = Number(p.get('L') ?? '128');
    const limits = (p.get('limits') ?? 'minimum') as 'minimum' | 'default';
    const inputs = await loadInputs(slug, L);
    const enc = await EncoderModel.load({
      manifestUrl: `/models/k28/${slug}/f16/manifest.json`, precision: 'f16', buckets: [L], limits,
    });
    result.info = enc.info();
    // Internals reached on purpose (diagnosis page): the bucket's executor and the input packing of runIds.
    const anyEnc = enc as unknown as {
      plans: Map<number, { profileForward: (i: unknown, g: 'dispatch', s: number) => Promise<{ times: Record<string, number> } | null> }>;
      kh: { device: GPUDevice };
      maskAndTypes: (i: unknown[], len: number) => { mask: Float32Array; typeIds?: Uint32Array };
      wordRows: (i: unknown[], len: number) => Float32Array | Uint16Array;
    };
    const lim: Record<string, number> = {};
    for (const k in anyEnc.kh.device.limits) lim[k] = (anyEnc.kh.device.limits as unknown as Record<string, number>)[k];
    result.deviceLimits = lim;
    const plan = anyEnc.plans.get(L);
    if (!plan) throw new Error('bucket missing');
    for (let i = 0; i < 20; i += 1) await enc.runIds({ inputIds: inputs.row(i) });
    const wall: number[] = [];
    for (let i = 0; i < 50; i += 1) {
      const t = performance.now();
      await enc.runIds({ inputIds: inputs.row(20 + i) });
      wall.push(performance.now() - t);
    }
    result.wallMedianMs = median([...wall].sort((a, b) => a - b));
    const per = new Map<string, number[]>();
    const sums: number[] = [];
    for (let i = 0; i < 20; i += 1) {
      const input = { inputIds: inputs.row(100 + i) };
      const { mask, typeIds } = anyEnc.maskAndTypes([input], L);
      const r = await plan.profileForward({ embeddings: anyEnc.wordRows([input], L), mask, typeIds }, 'dispatch', L);
      if (!r) throw new Error('no timestamp-query');
      let sum = 0;
      for (const [k, v] of Object.entries(r.times)) {
        if (!per.has(k)) per.set(k, []);
        (per.get(k) as number[]).push(v);
        sum += v;
      }
      sums.push(sum);
    }
    result.gpuSumMedianMs = median([...sums].sort((a, b) => a - b));
    result.ops = [...per.entries()].map(([name, v]) => ({ name, medianUs: median([...v].sort((a, b) => a - b)) * 1000 }));
    enc.dispose();
    result.stage = 'done';
    result.done = true;
  } catch (e) {
    result.stage = 'error';
    result.error = String(e);
    result.done = true;
  }
}

void main();
