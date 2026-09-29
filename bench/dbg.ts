// Timing breakdown: upload+submit, onSubmittedWorkDone (GPU exec done),
// mapAsync readback, plus per-call total — 60 iterations on one input.

import { Kleinhirn } from '../src/index.ts';
import type { SchemaInput } from '../src/tokenizer/schema.ts';

declare global {
  interface Window { khDbg?: Record<string, unknown> }
}

const out: Record<string, unknown> = { stage: 'boot' };
window.khDbg = out;

function med(a: number[]): number {
  const s = [...a].sort((x, y) => x - y);
  return s[s.length >> 1];
}

async function main(): Promise<void> {
  try {
    const kh = await Kleinhirn.load({
      manifestUrl: '/models/small-upstream/f16/manifest.json',
      buckets: [128], precision: 'auto',
      // Same input is timed repeatedly; the K16 cache would serve hits.
      cacheSize: 0,
    });
    const golden = await (await fetch(
      '/tests/golden/small-upstream/texts1000_l128k16.json')).json();
    const item = golden.items[0];
    const input: SchemaInput = {
      inputIds: Int32Array.from(item.input_ids),
      attentionMask: Int32Array.from(item.attention_mask),
      markerIndices: new Int32Array(16),
      markerMask: new Float32Array(16),
      markerGroups: new Int32Array(16),
      seqLen: item.seq_len,
    };
    input.markerIndices.set(item.marker_indices);
    input.markerMask.set(item.marker_mask);
    input.markerGroups.set(item.marker_groups);

    const plan = (kh as unknown as {
      plans: Map<number, {
        upload: (i: unknown) => void;
        submit: (c: boolean, s?: Set<number>) => void;
        readLogits: () => Promise<Float32Array>;
        device: GPUDevice;
      }>;
      queue: Promise<unknown>;
      packedMarkers: (i: SchemaInput, markers: number) => Uint32Array;
      maskOf: (i: SchemaInput, l: number) => Float32Array;
      embeddingRows: (i: SchemaInput, p: unknown) => Float32Array | Uint16Array;
      kh: { device: GPUDevice };
      runPrepared: (i: unknown) => Promise<Float32Array>;
    });
    const p = plan.plans.get(128)!;
    const dev = plan.kh.device;
    // JS-side costs outside the GPU pass.
    const js: Record<string, number> = {};
    const timeJs = async (name: string, fn: () => unknown): Promise<number> => {
      const ts: number[] = [];
      for (let i = 0; i < 60; i += 1) {
        const a = performance.now();
        fn();
        ts.push(performance.now() - a);
      }
      js[name] = med(ts.slice(5));
      return js[name];
    };
    await timeJs('embRows', () => plan.embeddingRows(input, p));
    await timeJs('maskOf', () => plan.maskOf(input, 128));
    await timeJs('packed', () => plan.packedMarkers(input, 16));
    {
      const ts: number[] = [];
      for (let i = 0; i < 30; i += 1) {
        const a = performance.now();
        await plan.runPrepared(input);
        ts.push(performance.now() - a);
      }
      js.runPrepared = med(ts.slice(5));
    }
    // Same call over all 1000 items, like the model-only bench loop.
    const inputs = golden.items.map((it: {
      input_ids: number[]; attention_mask: number[]; marker_indices: number[];
      marker_mask: number[]; marker_groups: number[]; seq_len: number;
    }) => {
      const i: SchemaInput = {
        inputIds: Int32Array.from(it.input_ids),
        attentionMask: Int32Array.from(it.attention_mask),
        markerIndices: new Int32Array(16),
        markerMask: new Float32Array(16),
        markerGroups: new Int32Array(16),
        seqLen: it.seq_len,
      };
      i.markerIndices.set(it.marker_indices);
      i.markerMask.set(it.marker_mask);
      i.markerGroups.set(it.marker_groups);
      return i;
    });
    const all: number[] = [];
    for (const i of inputs) {
      const a = performance.now();
      await plan.runPrepared(i);
      all.push(performance.now() - a);
    }
    js.runPreparedAll1000 = med(all);
    out.js = js;
    const spec = (plan as unknown as { spec: { layers: number } }).spec;
    void spec;
    // Per-kernel attribution: full pass vs pass with kernel i skipped.
    const names = ['qkv', 'attn', 'attnOut', 'lnA', 'ffn1', 'ffn2', 'lnF'];
    const runOne = async (skip?: Set<number>): Promise<number> => {
      const ts: number[] = [];
      for (let i = 0; i < 30; i += 1) {
        p.upload({
          embeddings: plan.embeddingRows(input, p),
          mask: plan.maskOf(input, 128),
          packedMarkers: plan.packedMarkers(input, 16),
        });
        const a = performance.now();
        p.submit(false, skip);
        await dev.queue.onSubmittedWorkDone();
        ts.push(performance.now() - a);
      }
      return med(ts.slice(5));
    };
    const full = await runOne();
    const perKernel: Record<string, number> = {};
    for (const [i, n] of names.entries()) {
      perKernel[n] = Math.round((full - await runOne(new Set([i]))) * 100) / 100;
    }
    out.perKernel = perKernel;
    out.full = full;
    const tSubmit: number[] = [];
    const tDone: number[] = [];
    const tMap: number[] = [];
    const tTotal: number[] = [];
    for (let i = 0; i < 60; i += 1) {
      const a = performance.now();
      p.upload({
        embeddings: plan.embeddingRows(input, p),
        mask: plan.maskOf(input, 128),
        packedMarkers: plan.packedMarkers(input, 16),
      });
      p.submit(false);
      const b = performance.now();
      await dev.queue.onSubmittedWorkDone();
      const c = performance.now();
      await p.readLogits();
      const d = performance.now();
      tSubmit.push(b - a);
      tDone.push(c - b);
      tMap.push(d - c);
      tTotal.push(d - a);
    }
    out.medians = {
      uploadSubmit: med(tSubmit.slice(10)),
      gpuDone: med(tDone.slice(10)),
      map: med(tMap.slice(10)),
      total: med(tTotal.slice(10)),
    };
    out.done = true;
  } catch (e) {
    out.error = String(e);
    out.done = true;
  }
}
void main();
