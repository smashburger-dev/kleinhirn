// K16 batch parity: every row of a GPU batch must reproduce the single-call
// result. GLiNER models draw rows from texts1000_l128k16 plus long200_l512
// for cross-bucket mixes; julia-1 uses the published parity100 requests.
// Reference is the single-input path on the SAME bucket the batch resolves
// to (batch rows and single rows run the same kernels and should be
// bitwise identical); golden argmax is reported alongside.
// Query: ?model=small-upstream|julia-1&precision=f32|f16&sizes=1,4,8,16
//        &limits=minimum|default

import { Kleinhirn } from '../src/index.ts';
import { JuliaEngine } from '../src/julia.ts';
import type { SchemaInput } from '../src/tokenizer/schema.ts';
import type { JuliaPreparedInput } from '../src/julia.ts';
import { argmax, compareLogits } from './metrics.ts';

const K_MAX = 16;

interface GoldenItem {
  seq_len: number;
  input_ids: number[];
  attention_mask: number[];
  marker_indices: number[];
  marker_mask: number[];
  marker_groups?: number[];
  logits: number[];
}

interface JuliaItem {
  seq_len: number;
  input_ids: number[];
  markers: number[];
  qtype: number;
  logits: number[];
}

interface BatchSet {
  batches: number;
  rows: number;
  vsSingle: Record<string, unknown>;
  vsGolden: Record<string, unknown>;
}

interface Result {
  stage: string;
  model?: string;
  precision?: string;
  info?: unknown;
  adapterInfo?: unknown;
  sizes?: Record<string, BatchSet>;
  mixed?: Record<string, BatchSet>;
  gates?: Record<string, boolean>;
  error?: string;
  done?: boolean;
}

declare global {
  interface Window { khBatchResult?: Result }
}

function toInput(item: GoldenItem): SchemaInput {
  const markerIndices = new Int32Array(K_MAX);
  markerIndices.set(item.marker_indices.slice(0, K_MAX));
  const markerMask = new Float32Array(K_MAX);
  markerMask.set(item.marker_mask.slice(0, K_MAX));
  const markerGroups = new Int32Array(K_MAX);
  markerGroups.set((item.marker_groups ?? []).slice(0, K_MAX));
  return {
    inputIds: Int32Array.from(item.input_ids),
    attentionMask: Int32Array.from(item.attention_mask),
    markerIndices, markerMask, markerGroups,
    seqLen: item.seq_len,
  };
}

function toJuliaInput(item: JuliaItem): JuliaPreparedInput {
  return {
    inputIds: Int32Array.from(item.input_ids), markers: item.markers,
    qtype: item.qtype, seqLen: item.seq_len,
  };
}

const chunksOf = <T>(arr: T[], n: number): T[][] => {
  const out: T[][] = [];
  for (let i = 0; i + n <= arr.length; i += n) out.push(arr.slice(i, i + n));
  return out;
};

async function glinerBatchParity(
  model: string,
  precision: string, limits: 'minimum' | 'default', sizes: number[],
  result: Result,
): Promise<void> {
  // cacheSize 0: the single runs must not seed the cache, or the batch
  // call would serve hits and the GPU batch path would never execute.
  const kh = await Kleinhirn.load({
    manifestUrl: `/models/${model}/${precision}/manifest.json`,
    buckets: [128, 512], precision: 'auto', limits, cacheSize: 0,
  });
  result.info = kh.info();
  result.adapterInfo = (result.info as { adapter?: unknown }).adapter;
  const golden = (await (await fetch(
    `/tests/golden/${model}/texts1000_l128k16.json`)).json()) as {
      items: GoldenItem[] };
  const long = (await (await fetch(
    `/tests/golden/${model}/long200_l512k16.json`)).json()) as {
      items: GoldenItem[] };
  const short = golden.items.slice(0, 64);
  const longItems = long.items;

  result.sizes = {};
  result.mixed = {};
  for (const B of sizes) {
    // Same-bucket batches at L128.
    const ref: number[][] = [];
    const cand: number[][] = [];
    const refG: number[][] = [];
    const candG: number[][] = [];
    const chunks = chunksOf(short, B);
    for (const chunk of chunks) {
      const singles = await Promise.all(chunk.map(
        (item) => kh.runPrepared(toInput(item), false, 128)));
      const batch = await kh.runPreparedBatch(
        chunk.map(toInput), 128);
      for (const [i, item] of chunk.entries()) {
        const nValid = item.marker_mask.filter((m) => m > 0.5).length;
        ref.push(Array.from(singles[i].logits.slice(0, nValid)));
        cand.push(Array.from(batch[i].logits.slice(0, nValid)));
        refG.push(item.logits.slice(0, nValid));
        candG.push(Array.from(batch[i].logits.slice(0, nValid)));
      }
    }
    result.sizes[`B${B}`] = {
      batches: chunks.length, rows: ref.length,
      vsSingle: compareLogits(ref, cand) as unknown as Record<string, unknown>,
      vsGolden: compareLogits(refG, candG) as unknown as Record<string, unknown>,
    };
    result.stage = `B${B} done`;

    // Mixed-length batches: short and long rows share the L512 bucket.
    const mixRef: number[][] = [];
    const mixCand: number[][] = [];
    const mixRefG: number[][] = [];
    const mixCandG: number[][] = [];
    const pool: GoldenItem[] = [];
    for (let i = 0; i < B * 4; i += 1) {
      pool.push(i % 2 === 0
        ? short[i % short.length]
        : longItems[i % longItems.length]);
    }
    for (const chunk of chunksOf(pool, B)) {
      const singles = await Promise.all(chunk.map(
        (item) => kh.runPrepared(toInput(item), false, 512)));
      const batch = await kh.runPreparedBatch(chunk.map(toInput), 512);
      for (const [i, item] of chunk.entries()) {
        const nValid = item.marker_mask.filter((m) => m > 0.5).length;
        mixRef.push(Array.from(singles[i].logits.slice(0, nValid)));
        mixCand.push(Array.from(batch[i].logits.slice(0, nValid)));
        mixRefG.push(item.logits.slice(0, nValid));
        mixCandG.push(Array.from(batch[i].logits.slice(0, nValid)));
      }
    }
    result.mixed[`B${B}`] = {
      batches: Math.floor(pool.length / B), rows: mixRef.length,
      vsSingle: compareLogits(mixRef, mixCand) as unknown as Record<string, unknown>,
      vsGolden: compareLogits(mixRefG, mixCandG) as unknown as Record<string, unknown>,
    };
    result.stage = `mixed B${B} done`;
  }
  kh.dispose();
}

async function juliaBatchParity(
  precision: string, limits: 'minimum' | 'default', sizes: number[],
  result: Result,
): Promise<void> {
  const kh = await JuliaEngine.load({
    manifestUrl: `/models/julia-1/${precision}/manifest.json`,
    buckets: [512, 1024], precision: 'auto', limits, cacheSize: 0,
  });
  result.info = kh.info();
  result.adapterInfo = (result.info as { adapter?: unknown }).adapter;
  const golden = (await (await fetch(
    '/tests/golden/julia-1/parity100.json')).json()) as { items: JuliaItem[] };
  const items = golden.items;

  result.sizes = {};
  result.mixed = {};
  for (const B of sizes) {
    const ref: number[][] = [];
    const cand: number[][] = [];
    const refG: number[][] = [];
    const candG: number[][] = [];
    // parity100 items already mix lengths; consecutive chunks of B cover
    // the mixed-length case, bucketed by each chunk's longest row.
    const chunks = chunksOf(items.slice(0, 64), B);
    for (const chunk of chunks) {
      const maxSeq = Math.max(...chunk.map((c) => c.seq_len));
      const bucket = maxSeq <= 512 ? 512 : 1024;
      const singles = await Promise.all(chunk.map(
        (item) => kh.runPrepared(toJuliaInput(item), false, bucket)));
      const batch = await kh.runPreparedBatch(
        chunk.map(toJuliaInput), bucket);
      for (const [i, item] of chunk.entries()) {
        const n = item.markers.length;
        ref.push(Array.from(singles[i].logits.slice(0, n)));
        cand.push(Array.from(batch[i].logits.slice(0, n)));
        refG.push(item.logits.slice(0, n));
        candG.push(Array.from(batch[i].logits.slice(0, n)));
      }
    }
    result.sizes[`B${B}`] = {
      batches: chunks.length, rows: ref.length,
      vsSingle: compareLogits(ref, cand) as unknown as Record<string, unknown>,
      vsGolden: compareLogits(refG, candG) as unknown as Record<string, unknown>,
    };
    result.stage = `B${B} done`;
  }
  kh.dispose();
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const model = params.get('model') ?? 'small-upstream';
  const precision = params.get('precision') ?? 'f32';
  const limits = params.get('limits') === 'default' ? 'default' : 'minimum';
  const sizes = params.get('sizes')?.split(',').map(Number) ?? [1, 4, 8, 16];
  const result: Result = { stage: 'boot', model, precision };
  window.khBatchResult = result;
  try {
    if (model === 'julia-1') {
      await juliaBatchParity(precision, limits, sizes, result);
    } else {
      await glinerBatchParity(model, precision, limits, sizes, result);
    }
    // f32: identical to the single call (<= 1e-5) and 100 % argmax.
    // f16: 100 % argmax against the f16 single-call path.
    const gates: Record<string, boolean> = {};
    const all = {
      ...(result.sizes ?? {}),
      ...Object.fromEntries(Object.entries(result.mixed ?? {})
        .map(([k, v]) => [`mixed-${k}`, v])),
    };
    for (const [name, s] of Object.entries(all)) {
      const vs = s.vsSingle as {
        argmaxAgreement: number; maxAbsLogitDiff: number };
      const vg = s.vsGolden as { argmaxAgreement: number };
      if (precision === 'f32') {
        gates[`${name}-logits`] = vs.maxAbsLogitDiff <= 1e-5;
        gates[`${name}-argmax`] = vs.argmaxAgreement === 1;
        gates[`${name}-golden`] = vg.argmaxAgreement === 1;
      } else {
        gates[`${name}-argmax`] = vs.argmaxAgreement === 1;
        gates[`${name}-golden`] = vg.argmaxAgreement === 1;
      }
    }
    result.gates = gates;
    result.stage = 'done';
    result.done = true;
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
  }
}

void main();
