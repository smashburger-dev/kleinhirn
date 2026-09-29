// K16 batch latency bench: wall time per decision inside a GPU batch of
// B = 1|4|8|16 for small-upstream (golden inputs at one bucket) and
// julia-1 (the 100 published requests). Measurement boundary: before
// runPreparedBatch/decideBatch to probabilities materialized in JS, after
// a warm pass; the cache is disabled so every timed batch computes.
// Query: ?model=small-upstream|julia-1&precision=f16|f32&batch=1|4|8|16
//        &bucket=128&limits=minimum|default

import { Kleinhirn } from '../src/index.ts';
import { JuliaEngine } from '../src/julia.ts';
import type { SchemaInput } from '../src/tokenizer/schema.ts';
import type { JuliaPreparedInput } from '../src/julia.ts';
import type { JuliaRequest } from '../src/tokenizer/julia-input.ts';
import { argmax, summarizeLatency } from './metrics.ts';

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

interface JuliaCase {
  request: JuliaRequest;
  pytorch_logits: number[];
}

interface Out {
  stage: string;
  model?: string;
  precision?: string;
  batch?: number;
  info?: unknown;
  adapterInfo?: unknown;
  loadMs?: number;
  result?: unknown;
  error?: string;
  done?: boolean;
}

declare global {
  interface Window { khBatchBench?: Out }
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

const chunksOf = <T>(arr: T[], n: number): T[][] => {
  const out: T[][] = [];
  for (let i = 0; i + n <= arr.length; i += n) out.push(arr.slice(i, i + n));
  return out;
};

async function glinerBench(
  precision: string, limits: 'minimum' | 'default',
  B: number, bucket: number, out: Out,
): Promise<void> {
  const tLoad = performance.now();
  const kh = await Kleinhirn.load({
    manifestUrl: `/models/small-upstream/${precision}/manifest.json`,
    buckets: [bucket], precision: 'auto', limits,
    cacheSize: 0, // timed runs must always reach the GPU
  });
  out.loadMs = performance.now() - tLoad;
  out.info = kh.info();
  out.adapterInfo = (out.info as { adapter?: unknown }).adapter;
  const golden = (await (await fetch(
    `/tests/golden/small-upstream/${
      bucket === 128 ? 'texts1000_l128k16' : `long200_l${bucket}k16`}.json`))
    .json()) as { items: GoldenItem[] };
  const batches = chunksOf(golden.items, B);
  const inputs = batches.map((c) => c.map(toInput));

  out.stage = 'warmup';
  for (const chunk of inputs.slice(0, 4)) {
    await kh.runPreparedBatch(chunk, bucket);
  }

  out.stage = 'timed';
  const perDecision: number[] = [];
  const perBatch: number[] = [];
  let argmaxHits = 0;
  let argmaxTotal = 0;
  for (const [i, chunk] of inputs.entries()) {
    const t = performance.now();
    const res = await kh.runPreparedBatch(chunk, bucket);
    res[0].probabilities[0]; // boundary: results materialized
    const ms = performance.now() - t;
    perBatch.push(ms);
    perDecision.push(ms / chunk.length);
    // Parity association: argmax of every timed row against its golden.
    for (const [j, item] of batches[i].entries()) {
      const nValid = item.marker_mask.filter((m) => m > 0.5).length;
      argmaxHits += Number(
        argmax(Array.from(res[j].logits.slice(0, nValid)))
          === argmax(item.logits.slice(0, nValid)));
      argmaxTotal += 1;
    }
    if (i % 10 === 0) out.stage = `timed ${i}/${inputs.length}`;
  }
  out.result = {
    batch_size: B, n_batches: inputs.length, n_requests: perDecision.length,
    batch_ms: summarizeLatency(perBatch),
    ms_per_decision: summarizeLatency(perDecision).medianMs,
    per_decision: summarizeLatency(perDecision),
    parity: { argmaxAgreement: argmaxHits / argmaxTotal, n: argmaxTotal },
  };
  kh.dispose();
}

async function juliaBench(
  precision: string, limits: 'minimum' | 'default',
  B: number, out: Out,
): Promise<void> {
  const tLoad = performance.now();
  const kh = await JuliaEngine.load({
    manifestUrl: `/models/julia-1/${precision}/manifest.json`,
    buckets: [512, 1024], precision: 'auto', limits,
    cacheSize: 0,
  });
  out.loadMs = performance.now() - tLoad;
  out.info = kh.info();
  out.adapterInfo = (out.info as { adapter?: unknown }).adapter;
  const cases = (await (await fetch(
    '/models/julia-1/onnx/parity-cases.json')).json()) as JuliaCase[];
  const prepared: JuliaPreparedInput[] = cases.map((c) => {
    const input = kh.prepare(c.request);
    return {
      inputIds: input.inputIds, markers: input.markers,
      qtype: input.qtype, seqLen: input.seqLen,
    };
  });
  const batches = chunksOf(prepared, B);

  out.stage = 'warmup';
  for (const chunk of batches.slice(0, 2)) {
    await kh.runPreparedBatch(chunk);
  }

  out.stage = 'timed';
  const perDecision: number[] = [];
  const perBatch: number[] = [];
  let argmaxHits = 0;
  let argmaxTotal = 0;
  for (const [i, chunk] of batches.entries()) {
    const t = performance.now();
    const res = await kh.runPreparedBatch(chunk);
    res[0].probabilities[0];
    const ms = performance.now() - t;
    perBatch.push(ms);
    perDecision.push(ms / chunk.length);
    for (const [j, input] of chunk.entries()) {
      const n = input.markers.length;
      argmaxHits += Number(
        argmax(Array.from(res[j].logits.slice(0, n)))
          === argmax(cases[i * B + j].pytorch_logits));
      argmaxTotal += 1;
    }
    if (i % 10 === 0) out.stage = `timed ${i}/${batches.length}`;
  }
  out.result = {
    batch_size: B, n_batches: batches.length,
    n_requests: perDecision.length,
    batch_ms: summarizeLatency(perBatch),
    ms_per_decision: summarizeLatency(perDecision).medianMs,
    per_decision: summarizeLatency(perDecision),
    parity: { argmaxAgreement: argmaxHits / argmaxTotal, n: argmaxTotal },
  };
  kh.dispose();
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const model = params.get('model') ?? 'small-upstream';
  const precision = params.get('precision') ?? 'f16';
  const limits = params.get('limits') === 'default' ? 'default' : 'minimum';
  const B = Number(params.get('batch') ?? '4');
  const bucket = Number(params.get('bucket') ?? '128');
  const out: Out = { stage: 'boot', model, precision, batch: B };
  window.khBatchBench = out;
  try {
    if (model === 'julia-1') {
      await juliaBench(precision, limits, B, out);
    } else {
      await glinerBench(precision, limits, B, bucket, out);
    }
    out.stage = 'done';
    out.done = true;
  } catch (error) {
    out.stage = 'error';
    out.error = String(error);
    out.done = true;
  }
}

void main();
