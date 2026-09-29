// kleinhirn parity bench: layer states vs golden bins (f32 capture only),
// logits on all corpus items and the two-task cases, per docs/PLAN.md K3 B/C.
// Query: ?model=small-upstream|base-upstream|multi-upstream&precision=f32|f16
// Layer-capture positions are compared on valid rows (row < seq_len) only.

import { Kleinhirn } from '../src/index.ts';
import { WasmKleinhirn } from '../src/wasm.ts';
import type { PreparedResult } from '../src/index.ts';
import type { SchemaInput } from '../src/tokenizer/schema.ts';
import { argmax, compareLogits } from './metrics.ts';

interface Engine {
  runPrepared(input: SchemaInput, capture?: boolean, bucket?: number): Promise<PreparedResult>;
  info(): Record<string, unknown>;
  dispose(): void;
}

const K_MAX = 16;
// Hidden width comes from the loaded manifest (384 small, 768 base/multi);
// layer golden tensors are laid out L x hidden.
let HIDDEN = 384;

interface TensorRef { offset: number; shape: number[]; dtype: string }

interface BucketCase {
  seq_len: number;
  input_ids: number[];
  attention_mask: number[];
  marker_indices: number[];
  marker_mask: number[];
  marker_groups: number[];
  logits: number[];
  probabilities: number[];
  label_index: number;
  tensors: Record<string, TensorRef>;
}

interface LayerIndex {
  tensor_order: string[];
  cases: { title: string; buckets: Record<string, BucketCase> }[];
}

interface GoldenItem {
  seq_len: number;
  input_ids: number[];
  attention_mask: number[];
  marker_indices: number[];
  marker_mask: number[];
  marker_groups?: number[];
  logits: number[];
}

interface TwoTaskItem {
  seq_len: number;
  input_ids: number[];
  attention_mask: number[];
  marker_indices: number[];
  marker_mask: number[];
  marker_groups: number[];
  tasks: { task: string; logits: number[]; label_index: number }[];
}

interface ParityResult {
  stage: string;
  model?: string;
  precision?: string;
  backend?: string;
  adapterInfo?: unknown;
  info?: unknown;
  layers?: unknown;
  longLayers?: unknown;
  corpus?: unknown;
  twoTasks?: unknown;
  long?: Record<string, unknown>;
  gates?: Record<string, boolean>;
  error?: string;
  done?: boolean;
}

declare global {
  interface Window { khParityResult?: ParityResult }
}

function toInput(b: {
  seq_len: number; input_ids: number[]; attention_mask: number[];
  marker_indices: number[]; marker_mask: number[]; marker_groups?: number[];
}): SchemaInput {
  const markerIndices = new Int32Array(K_MAX);
  markerIndices.set(b.marker_indices.slice(0, K_MAX));
  const markerMask = new Float32Array(K_MAX);
  markerMask.set(b.marker_mask.slice(0, K_MAX));
  const markerGroups = new Int32Array(K_MAX);
  markerGroups.set((b.marker_groups ?? []).slice(0, K_MAX));
  return {
    inputIds: Int32Array.from(b.input_ids),
    attentionMask: Int32Array.from(b.attention_mask),
    markerIndices, markerMask, markerGroups,
    seqLen: b.seq_len,
  };
}

// Valid marker logits in group order, matching the golden layout.
function validLogits(
  logits: Float32Array, groups: Int32Array, mask: Float32Array, group: number,
): number[] {
  const out: number[] = [];
  for (let k = 0; k < groups.length; k += 1) {
    if (mask[k] > 0.5 && groups[k] === group) out.push(logits[k]);
  }
  return out;
}

async function layerParity(
  kh: Engine, model: string, result: ParityResult, indexName = 'layers',
): Promise<Record<string, unknown>> {
  const idx = (await (await fetch(
    `/models/${model}/golden/${indexName}.index.json`)).json()) as LayerIndex;
  const blob = await (await fetch(
    `/models/${model}/golden/${indexName}.bin`)).arrayBuffer();
  const order = idx.tensor_order;
  let maxTensorDiff = 0;
  let maxLogitDiff = 0;
  let agree = 0;
  let n = 0;
  const perTensor: Record<string, number> = {};
  for (const name of order) perTensor[name] = 0;
  for (const [ci, c] of idx.cases.entries()) {
    for (const [bk, b] of Object.entries(c.buckets)) {
      if (!b) continue; // cases over 128 tokens only have L256
      const L = Number(bk.slice(1));
      const res = await kh.runPrepared(toInput(b), true, L);
      const cap = res.captureData;
      if (!cap) throw new Error('capture missing (f32 only)');
      for (const [s, name] of order.entries()) {
        const t = b.tensors[name];
        const golden = new Float32Array(blob, t.offset, L * HIDDEN);
        const got = cap.subarray(s * L * HIDDEN, (s + 1) * L * HIDDEN);
        let mx = 0;
        for (let i = 0; i < b.seq_len * HIDDEN; i += 1) {
          const d = Math.abs(golden[i] - got[i]);
          if (d > mx) mx = d;
        }
        perTensor[name] = Math.max(perTensor[name], mx);
        maxTensorDiff = Math.max(maxTensorDiff, mx);
      }
      const nValid = b.marker_mask.filter((m) => m > 0.5).length;
      const cand = Array.from(res.logits.slice(0, nValid));
      for (let i = 0; i < nValid; i += 1) {
        maxLogitDiff = Math.max(maxLogitDiff, Math.abs(b.logits[i] - cand[i]));
      }
      if (argmax(b.logits) === argmax(cand)) agree += 1;
      n += 1;
    }
    result.stage = `${indexName} ${ci + 1}/${idx.cases.length}`;
  }
  return {
    n, maxTensorDiff, maxLogitDiff,
    argmaxAgreement: agree / n, perTensor,
  };
}

async function corpusParity(
  kh: Engine, model: string, result: ParityResult,
): Promise<Record<string, unknown>> {
  const golden = (await (await fetch(
    `/tests/golden/${model}/texts1000_l128k16.json`)).json()) as {
      items: GoldenItem[];
    };
  const ref: number[][] = [];
  const cand: number[][] = [];
  for (const [i, item] of golden.items.entries()) {
    const res = await kh.runPrepared(toInput(item));
    const nValid = item.marker_mask.filter((m) => m > 0.5).length;
    ref.push(item.logits.slice(0, nValid));
    cand.push(Array.from(res.logits.slice(0, nValid)));
    if (i % 100 === 0) result.stage = `corpus ${i}/${golden.items.length}`;
  }
  return compareLogits(ref, cand) as unknown as Record<string, unknown>;
}

async function longBucketParity(
  kh: Engine, model: string, length: number, result: ParityResult,
): Promise<Record<string, unknown>> {
  const golden = (await (await fetch(
    `/tests/golden/${model}/long200_l${length}k16.json`)).json()) as {
      items: GoldenItem[];
    };
  const ref: number[][] = [];
  const cand: number[][] = [];
  for (const [i, item] of golden.items.entries()) {
    const res = await kh.runPrepared(toInput(item), false, length);
    const nValid = item.marker_mask.filter((m) => m > 0.5).length;
    ref.push(item.logits.slice(0, nValid));
    cand.push(Array.from(res.logits.slice(0, nValid)));
    if (i % 40 === 0) result.stage = `long L${length} ${i}/${golden.items.length}`;
  }
  return compareLogits(ref, cand) as unknown as Record<string, unknown>;
}

async function twoTaskParity(
  kh: Engine, model: string,
): Promise<Record<string, unknown>> {
  const golden = (await (await fetch(
    `/tests/golden/${model}/two_tasks_l128k16.json`)).json()) as {
      items: TwoTaskItem[];
    };
  const ref: number[][] = [];
  const cand: number[][] = [];
  for (const item of golden.items) {
    const input = toInput(item);
    const res = await kh.runPrepared(input);
    for (const [g, t] of item.tasks.entries()) {
      ref.push(t.logits);
      cand.push(validLogits(res.logits, input.markerGroups, input.markerMask, g));
    }
  }
  return compareLogits(ref, cand) as unknown as Record<string, unknown>;
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const model = params.get('model') ?? 'small-upstream';
  const precision = params.get('precision') ?? 'f32';
  const limits = params.get('limits') === 'default' ? 'default' : 'minimum';
  const backend = params.get('backend') ?? 'webgpu';
  const result: ParityResult = { stage: 'boot', model, precision, backend };
  window.khParityResult = result;
  try {
    const t0 = performance.now();
    const buckets = params.get('buckets')?.split(',').map(Number) ?? [128, 256];
    const sets = params.get('sets') ?? 'all';
    const kh: Engine = backend === 'wasm'
      ? await WasmKleinhirn.load({
        manifestUrl: `/models/${model}/f32/manifest.json`,
      })
      : await Kleinhirn.load({
        manifestUrl: `/models/${model}/${precision}/manifest.json`,
        buckets,
        precision: 'auto',
        limits,
      });
    const mf = await (await fetch(
      `/models/${model}/${precision}/manifest.json`)).json() as {
      encoder?: { hiddenSize?: number } };
    HIDDEN = mf.encoder?.hiddenSize ?? 384;
    result.info = { ...kh.info(), loadMs: performance.now() - t0 };
    result.adapterInfo = (result.info as { adapter?: unknown }).adapter;

    if (precision === 'f32' && sets === 'all' && backend === 'webgpu') {
      result.stage = 'layers';
      result.layers = await layerParity(kh, model, result);
    }
    if (sets === 'all') {
      result.stage = 'corpus';
      result.corpus = await corpusParity(kh, model, result);
      result.stage = 'two-tasks';
      result.twoTasks = await twoTaskParity(kh, model);
    }
    const longBuckets = buckets.filter((b) => b >= 512);
    if (precision === 'f32' && longBuckets.length && backend === 'webgpu') {
      result.longLayers = await layerParity(kh, model, result, 'layers_long');
    }
    const long: Record<string, unknown> = {};
    for (const b of longBuckets) {
      long[`L${b}`] = await longBucketParity(kh, model, b, result);
    }
    if (longBuckets.length) result.long = long;

    const corpusSet = result.corpus as { argmaxAgreement: number; maxAbsLogitDiff: number; maxAbsProbDiff: number } | undefined;
    const two = result.twoTasks as { argmaxAgreement: number; maxAbsLogitDiff: number } | undefined;
    const layers = result.layers as { maxTensorDiff: number } | undefined;
    const longLayers = result.longLayers as { maxTensorDiff: number } | undefined;
    const longCmp = Object.values(long) as {
      argmaxAgreement: number; maxAbsLogitDiff: number; maxAbsProbDiff: number;
    }[];
    const longOk = (cmp: (c: { argmaxAgreement: number; maxAbsLogitDiff: number; maxAbsProbDiff: number }) => boolean) =>
      longCmp.every(cmp);
    result.gates = backend === 'wasm'
      ? {
        // K5 WASM gate: 100 % argmax, max logit deviation <= 1e-3 on the
        // 1000 corpus texts (plus two-task as extra coverage).
        corpusArgmax: sets === 'long' || corpusSet!.argmaxAgreement === 1,
        corpusLogits: sets === 'long' || corpusSet!.maxAbsLogitDiff <= 1e-3,
        twoTaskArgmax: sets === 'long' || two!.argmaxAgreement === 1,
        twoTaskLogits: sets === 'long' || two!.maxAbsLogitDiff <= 1e-3,
        longArgmax: !longCmp.length || longOk((c) => c.argmaxAgreement === 1),
        longLogits: !longCmp.length || longOk((c) => c.maxAbsLogitDiff <= 1e-3),
      }
      : precision === 'f32'
      ? {
        layerTensors: sets === 'long' || (layers?.maxTensorDiff ?? Infinity) <= 1e-3,
        longLayerTensors: !longBuckets.length
          || (longLayers?.maxTensorDiff ?? Infinity) <= 1e-3,
        corpusArgmax: sets === 'long' || corpusSet!.argmaxAgreement === 1,
        corpusLogits: sets === 'long' || corpusSet!.maxAbsLogitDiff <= 1e-3,
        twoTaskArgmax: sets === 'long' || two!.argmaxAgreement === 1,
        twoTaskLogits: sets === 'long' || two!.maxAbsLogitDiff <= 1e-3,
        longArgmax: !longCmp.length || longOk((c) => c.argmaxAgreement === 1),
        longLogits: !longCmp.length
          || longOk((c) => c.maxAbsLogitDiff <= 1e-3),
      }
      : {
        corpusArgmax: sets === 'long' || corpusSet!.argmaxAgreement >= 0.995,
        corpusProbs: sets === 'long' || corpusSet!.maxAbsProbDiff <= 1e-2,
        twoTaskArgmax: sets === 'long' || two!.argmaxAgreement >= 0.995,
        twoTaskProbs: sets === 'long'
          || (result.twoTasks as { maxAbsProbDiff: number }).maxAbsProbDiff <= 1e-2,
        longArgmax: !longCmp.length || longOk((c) => c.argmaxAgreement >= 0.995),
        longProbs: !longCmp.length || longOk((c) => c.maxAbsProbDiff <= 1e-2),
      };
    result.stage = 'done';
    result.done = true;
    kh.dispose();
  } catch (error) {
    result.stage = 'error';
    result.error = String(error);
    result.done = true;
  }
}

void main();
